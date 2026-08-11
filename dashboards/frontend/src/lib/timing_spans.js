import {
  CROSS_LANE_CONTAINMENT_TOLERANCE_S,
  CATEGORIES,
  IDLE_PHASES,
  NESTS_IN,
  NEGLIGIBLE_WORK_S,
  SAMPLED,
  TOOLTIP_HIDDEN_PHASES,
  categoryOf,
  labelFor,
  rolloutIdForTimingKey,
} from "./timing_vocabulary.js";

export function collect(timings) {
  const spans = [];
  for (const [id, lanes] of Object.entries(timings || {})) {
    const rolloutId = rolloutIdForTimingKey(id);
    for (const [role, lane] of Object.entries(lanes?.roles || {})) {
      if (lane?.lane_start_unix_s == null) continue;
      const laneStart = Number(lane?.lane_start_unix_s);
      if (!Number.isFinite(laneStart)) continue;
      for (const [name, phase] of Object.entries(lane?.phases || {})) {
        const count = Number(phase?.count) || 0;
        const total = Number(phase?.total_duration_s) || 0;
        if (!count || total < NEGLIGIBLE_WORK_S) continue;
        const where = {
          rolloutId,
          role,
          group: role === "rollout" ? "generation" : "step",
          name,
          category: categoryOf(name),
        };
        const invocations = Array.isArray(phase?.invocations)
          ? phase.invocations
          : [];
        const runs =
          !SAMPLED.has(name) && invocations.length === count
            ? invocations
            : [];
        if (runs.length) {
          for (const [from, to] of runs) {
            const start = laneStart + (Number(from) || 0);
            const end = laneStart + (Number(to) || 0);
            spans.push({
              ...where,
              laneKey: `${id}:${role}`,
              kind: IDLE_PHASES.has(name) ? "idle" : "work",
              start,
              end,
              count: 1,
              total: end - start,
              longest: end - start,
            });
          }
          continue;
        }
        spans.push({
          ...where,
          laneKey: `${id}:${role}`,
          kind: IDLE_PHASES.has(name)
            ? "idle"
            : SAMPLED.has(name) || count > 1
              ? "sampled"
              : "work",
          start: laneStart + (Number(phase?.first_start_s) || 0),
          end: laneStart + (Number(phase?.last_end_s) || 0),
          count,
          total,
          longest: Number(phase?.longest_duration_s) || 0,
        });
      }
    }
  }
  return anchorLanes(spans);
}

export function anchorLanes(spans) {
  const lanes = new Map();
  const drivers = new Map();
  for (const span of spans) {
    const key = `${span.rolloutId}:${span.role}`;
    const bucket = lanes.get(key) || [];
    bucket.push(span);
    lanes.set(key, bucket);
    if (span.role === "driver" && span.rolloutId != null) {
      const phases = drivers.get(span.rolloutId) || new Map();
      const bounds = phases.get(span.name);
      phases.set(
        span.name,
        bounds
          ? {
              start: Math.min(bounds.start, span.start),
              end: Math.max(bounds.end, span.end),
            }
          : { start: span.start, end: span.end },
      );
      drivers.set(span.rolloutId, phases);
    }
  }

  for (const lane of lanes.values()) {
    const first = lane[0];
    if (!first || first.role === "driver" || first.rolloutId == null) continue;
    const categories = new Set(lane.map((span) => span.category));
    if (categories.size !== 1) continue;
    const category = CATEGORIES[[...categories][0]];
    if (!category?.owner) continue;
    const parent = drivers.get(first.rolloutId)?.get(category.owner);
    if (!parent) continue;

    let laneStart = Infinity;
    let laneEnd = -Infinity;
    for (const span of lane) {
      laneStart = Math.min(laneStart, span.start);
      laneEnd = Math.max(laneEnd, span.end);
    }
    let offset = 0;
    if (laneEnd > parent.end) offset = parent.end - laneEnd;
    if (laneStart + offset < parent.start) offset = parent.start - laneStart;
    if (Math.abs(offset) < 1e-9) continue;

    for (const span of lane) {
      span.start += offset;
      span.end += offset;
      span.clockShifted = true;
      span.clockOffset = offset;
    }
  }
  return spans;
}

export function timingIsAsync(timings) {
  const generations = [];
  const steps = [];
  let sync = false;
  for (const lanes of Object.values(timings || {})) {
    for (const [role, lane] of Object.entries(lanes?.roles || {})) {
      const laneStart = Number(lane?.lane_start_unix_s);
      if (!Number.isFinite(laneStart)) continue;
      for (const [name, phase] of Object.entries(lane?.phases || {})) {
        const count = Number(phase?.count) || 0;
        const total = Number(phase?.total_duration_s) || 0;
        if (!count || total < NEGLIGIBLE_WORK_S) continue;
        const start = laneStart + Number(phase?.first_start_s);
        const end = laneStart + Number(phase?.last_end_s);
        if (!Number.isFinite(start) || !Number.isFinite(end)) continue;
        if (role === "rollout") {
          generations.push([start, end]);
        } else if (role === "driver" && name === "generate_rollouts") {
          sync = true;
        } else if (!IDLE_PHASES.has(name)) {
          steps.push([start, end]);
        }
      }
    }
  }
  return (
    !sync &&
    generations.some(([generationStart, generationEnd]) =>
      steps.some(
        ([stepStart, stepEnd]) =>
          generationStart < stepEnd && stepStart < generationEnd,
      ),
    )
  );
}

export function groupTooltipChildren(children) {
  const groups = new Map();
  for (const child of children || []) {
    if (child.mergedGeneration || TOOLTIP_HIDDEN_PHASES.has(child.name)) {
      continue;
    }
    const count = child.count || 1;
    const group = groups.get(child.name);
    if (group) {
      group.duration += child.duration;
      group.count += count;
      continue;
    }
    groups.set(child.name, {
      name: child.name,
      label: labelFor(child.name, child.rolloutId),
      duration: child.duration,
      count,
      representative: child,
    });
  }
  return [...groups.values()];
}

export function stepsOf(spans) {
  const byRollout = new Map();
  for (const span of spans) {
    if (span.role !== "driver" || span.rolloutId == null) continue;
    const step = byRollout.get(span.rolloutId) ?? {
      id: span.rolloutId,
      start: span.start,
      end: span.end,
      work: 0,
      idle: 0,
    };
    step.start = Math.min(step.start, span.start);
    step.end = Math.max(step.end, span.end);
    if (span.kind === "idle") step.idle += span.total;
    else step.work += span.total;
    byRollout.set(span.rolloutId, step);
  }
  return [...byRollout.values()].sort((a, b) => a.start - b.start);
}

export function nest(spans) {
  const ordered = [...spans].sort((a, b) => a.start - b.start || b.end - a.end);
  const byRolloutAndName = new Map();
  for (const [index, span] of ordered.entries()) {
    span.orderIndex = index;
    span.depth = 0;
    span.children = [];
    const key = `${span.rolloutId}:${span.name}`;
    const bucket = byRolloutAndName.get(key) || [];
    bucket.push(span);
    byRolloutAndName.set(key, bucket);
  }
  for (const span of ordered) {
    let parent = null;
    for (const parentName of NESTS_IN[span.name] || []) {
      for (const other of byRolloutAndName.get(
        `${span.rolloutId}:${parentName}`,
      ) || []) {
        if (
          other !== span &&
          (other.group === span.group ||
            (span.name === "generate_samples" &&
              other.name === "generate_rollouts")) &&
          (other.laneKey === span.laneKey
            ? other.start <= span.start && span.end <= other.end
            : other.start <= span.start + CROSS_LANE_CONTAINMENT_TOLERANCE_S &&
              span.end <= other.end + CROSS_LANE_CONTAINMENT_TOLERANCE_S) &&
          (!parent ||
            other.depth > parent.depth ||
            (other.depth === parent.depth &&
              other.orderIndex < parent.orderIndex))
        ) {
          parent = other;
        }
      }
    }
    span.depth = parent ? parent.depth + 1 : 0;
    span.parent = parent ?? null;
    if (parent) parent.contains = true;
    if (parent) parent.children.push(span);
  }
  for (const span of ordered) {
    const occurrences = new Map();
    for (const child of [...span.children].sort((a, b) => a.start - b.start)) {
      if (!["forward_backward", "optimizer_step"].includes(child.name)) continue;
      const occurrence = (occurrences.get(child.name) || 0) + 1;
      occurrences.set(child.name, occurrence);
      child.ordinal = occurrence;
    }
    const duration = Math.max(span.end - span.start, 0);
    const aggregates = new Map();
    for (const child of span.children) {
      if (!SAMPLED.has(child.name)) {
        continue;
      }
      const childDuration = Math.max(child.end - child.start, 0);
      const current = aggregates.get(child.name) ?? {
        name: child.name,
        kind: child.kind,
        clockShifted: child.clockShifted,
        clockOffset: child.clockOffset,
        category: child.category,
        role: child.role,
        rolloutId: child.rolloutId,
        duration: 0,
        total: 0,
        count: 0,
        longest: 0,
        start: child.start,
        end: child.end,
      };
      current.duration += childDuration;
      current.total += child.total ?? childDuration;
      current.count += child.count || 1;
      current.longest = Math.max(current.longest, child.longest || childDuration);
      current.start = Math.min(current.start, child.start);
      current.end = Math.max(current.end, child.end);
      aggregates.set(child.name, current);
    }
    const children = span.children.filter(
      (child) => !SAMPLED.has(child.name),
    );
    for (const child of aggregates.values()) {
      children.push({
        ...child,
        share: duration > 0 ? child.duration / duration : 0,
        average: child.count ? child.total / child.count : 0,
      });
    }
    span.children = children;
    delete span.orderIndex;
  }
  return ordered;
}
