import {
  CATEGORIES,
  GROUPS,
  NEGLIGIBLE_WORK_S,
  TOOLTIP_HIDDEN_PHASES,
} from "./timing_vocabulary.js";
import { collect, isAsyncSpans, nest, stepsOf } from "./timing_spans.js";

function mergeSyncGenerationSpans(spans) {
  const drivers = new Map(
    spans
      .filter(
        (span) =>
          span.role === "driver" &&
          span.name === "generate_rollouts" &&
          span.rolloutId != null,
      )
      .map((span) => [span.rolloutId, span]),
  );
  for (const span of spans) {
    if (span.role !== "rollout" || span.name !== "generate_samples") continue;
    const driver = drivers.get(span.rolloutId);
    if (!driver) continue;
    const sampleGeneration = nestedChild(span, "sample_generation");
    const aggregateStats = { ...(driver.aggregateStats || {}) };
    const descendants = [...(span.children || [])];
    for (let index = 0; index < descendants.length; index += 1) {
      for (const child of descendants[index].children || []) {
        descendants.push(child);
      }
    }
    if (sampleGeneration) {
      aggregateStats.sample_generation = sampleGeneration;
    }
    for (const descendant of descendants) {
      if (!TOOLTIP_HIDDEN_PHASES.has(descendant.name)) {
        const duration = descendant.duration ?? descendant.end - descendant.start;
        const aggregate = {
          ...descendant,
          duration,
          total: descendant.total ?? duration,
          count: descendant.count ?? 1,
          longest: descendant.longest ?? duration,
          mergedGeneration: false,
        };
        const current = aggregateStats[descendant.name];
        if (current) {
          current.duration += aggregate.duration;
          current.total += aggregate.total;
          current.count += aggregate.count;
          current.longest = Math.max(current.longest, aggregate.longest);
          current.start = Math.min(current.start, aggregate.start);
          current.end = Math.max(current.end, aggregate.end);
        } else {
          aggregateStats[descendant.name] = aggregate;
        }
      }
    }
    if (Object.keys(aggregateStats).length) {
      driver.aggregateStats = aggregateStats;
    }
    span.mergedGeneration = true;
    while (descendants.length) {
      const descendant = descendants.pop();
      descendant.mergedGeneration = true;
      for (const child of descendant.children || []) {
        descendants.push(child);
      }
    }
    if (span.parent) {
      span.parent.children = span.parent.children.filter(
        (child) => child.name !== "generate_samples",
      );
    }
  }
  return spans;
}

function nestedChild(span, name) {
  for (const child of span.children || []) {
    if (child.name === name) return child;
    const nested = nestedChild(child, name);
    if (nested) return nested;
  }
  return null;
}

export function clipIdleSpans(spans, async) {
  const workByRow = new Map();
  for (const span of spans) {
    if (span.kind === "idle") continue;
    const row = async && span.role === "rollout" ? "generation" : "step";
    const intervals = workByRow.get(row) || [];
    intervals.push([span.start, span.end]);
    workByRow.set(row, intervals);
  }
  for (const [row, intervals] of workByRow.entries()) {
    intervals.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const merged = [];
    for (const [start, end] of intervals) {
      const previous = merged.at(-1);
      if (previous && start <= previous[1]) {
        previous[1] = Math.max(previous[1], end);
      } else {
        merged.push([start, end]);
      }
    }
    workByRow.set(row, merged);
  }
  const piecesByIndex = new Map();
  spans.forEach((span, index) => {
    if (span.kind !== "idle") piecesByIndex.set(index, [span]);
  });
  const idleSpans = spans
    .map((span, index) => ({ span, index }))
    .filter(({ span }) => span.kind === "idle")
    .sort((a, b) => a.span.start - b.span.start);
  for (const { span, index } of idleSpans) {
    const pieces = [];
    const row = async && span.role === "rollout" ? "generation" : "step";
    const ranges = workByRow.get(row) || [];
    let low = 0;
    let high = ranges.length;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (ranges[middle][1] <= span.start) low = middle + 1;
      else high = middle;
    }
    let cursor = span.start;
    for (let current = low; current < ranges.length; current += 1) {
      const [workStart, workEnd] = ranges[current];
      if (workStart >= span.end) break;
      if (workStart > cursor) {
        const end = Math.min(workStart, span.end);
        if (end - cursor >= NEGLIGIBLE_WORK_S) {
          pieces.push({
            ...span,
            start: cursor,
            end,
            total: end - cursor,
            duration: end - cursor,
          });
        }
      }
      cursor = Math.max(cursor, workEnd);
      if (cursor >= span.end) break;
    }
    if (cursor < span.end) {
      const start = cursor;
      const end = span.end;
      if (end - start >= NEGLIGIBLE_WORK_S) {
        pieces.push({
          ...span,
          start,
          end,
          total: end - start,
          duration: end - start,
        });
      }
    }
    piecesByIndex.set(index, pieces);
  }
  const clipped = [];
  for (let index = 0; index < spans.length; index += 1) {
    clipped.push(...(piecesByIndex.get(index) || []));
  }
  return clipped;
}

function rowsOf(spans, async) {
  if (!spans.length) return [];
  const prepareRow = (row) => {
    row.sortedSpans = [...row.spans].sort(
      (a, b) => a.depth - b.depth || a.start - b.start || b.end - a.end,
    );
    row.insetKeys = new Set();
    const byDepth = new Map();
    for (const span of row.sortedSpans) {
      const bucket = byDepth.get(span.depth) || [];
      bucket.push(span);
      byDepth.set(span.depth, bucket);
    }
    for (const spansAtDepth of byDepth.values()) {
      const ends = [...spansAtDepth].sort((a, b) => a.end - b.end);
      let endIndex = 0;
      let previousEnd = null;
      for (const span of spansAtDepth) {
        while (endIndex < ends.length && ends[endIndex].end <= span.start) {
          previousEnd = ends[endIndex].end;
          endIndex += 1;
        }
        if (previousEnd !== null && span.start > previousEnd) {
          row.insetKeys.add(span.key);
        }
      }
    }
    return row;
  };
  const driverSpans = spans.filter((span) => !async || span.role !== "rollout");
  if (!async) {
    return [prepareRow(
      {
        key: "driver",
        label: "Train",
        role: "driver",
        hint: "Driver and trainer phases on the shared wall clock.",
        spans: driverSpans,
      },
    )];
  }

  const rows = [
    prepareRow({
      key: "driver",
      label: "Train",
      role: "driver",
      hint: "Driver and trainer phases on the shared wall clock.",
      spans: driverSpans,
    }),
  ];
  const rolloutSpans = spans.filter((span) => span.role === "rollout");
  const roots = rolloutSpans.filter(
    (span) => span.depth === 0 && !span.mergedGeneration,
  );
  const packed = [];
  for (const span of [...roots].sort((a, b) => a.start - b.start || b.end - a.end)) {
    const row = packed.find((candidate) => candidate.end <= span.start);
    if (row) {
      row.end = span.end;
      row.roots.push(span);
    } else {
      packed.push({ end: span.end, roots: [span] });
    }
  }
  for (const [index, packedRow] of packed.entries()) {
    const rootSet = new Set(packedRow.roots);
    rows.push(prepareRow({
      key: `rollout-${index}`,
      label: "Rollouts",
      role: "rollout",
      hint: "Rollout engine phases packed by their actual wall-clock overlap.",
      spans: rolloutSpans.filter((span) => {
        let root = span;
        while (root.parent) root = root.parent;
        return rootSet.has(root);
      }),
    }));
  }
  return rows;
}

export function runTimeline(timings, asyncOverride = null) {
  const measured = collect(timings);
  if (!measured.length) {
    return { span: 0, runStart: null, async: false, groups: [], steps: [], categories: [] };
  }
  // Avoid spreading one argument per invocation; long runs can exceed the call stack.
  let runStart = Infinity;
  let runEnd = -Infinity;
  for (const span of measured) {
    runStart = Math.min(runStart, span.start);
    runEnd = Math.max(runEnd, span.end);
  }
  const steps = stepsOf(measured);
  const rawSpans = measured;
  const sync = rawSpans.some(
    (span) => span.role === "driver" && span.name === "generate_rollouts",
  );
  const async =
    asyncOverride ?? isAsyncSpans(rawSpans);
  const spans = nest(clipIdleSpans(rawSpans, async));
  if (sync) mergeSyncGenerationSpans(spans);

  for (const [index, span] of spans.entries()) {
    span.key = `${span.rolloutId}:${span.role}:${span.name}:${span.start.toFixed(3)}:${index}`;
  }
  for (const span of spans) {
    span.offset = span.start - runStart;
    span.duration = span.end - span.start;
    span.average = span.total / span.count;
    span.inside = span.parent ? span.parent.name : null;
    span.insideKey = span.parent ? span.parent.key : null;
    span.insideStart = span.parent ? span.parent.start : null;
    span.insideEnd = span.parent ? span.parent.end : null;
  }

  function hydrateNestedChildren(parent) {
    for (const child of parent.children || []) {
      child.offset = child.start - runStart;
      child.depth = parent.depth + 1;
      child.inside = parent.name;
      child.insideKey = parent.key;
      child.insideStart = parent.start;
      child.insideEnd = parent.end;
      child.key ??= `${parent.key}:child:${child.name}`;
      hydrateNestedChildren(child);
    }
  }
  for (const span of spans) {
    hydrateNestedChildren(span);
  }

  const visibleSpans = spans;
  const rows = rowsOf(visibleSpans, async);
  const groups = rows.length ? [{ ...GROUPS[0], rows }] : [];
  for (const span of spans) delete span.parent;

  return {
    runStart,
    span: Math.max(runEnd - runStart, 1e-6),
    async,
    groups,
    steps: steps.map((step) => ({
      ...step,
      offset: step.start - runStart,
      duration: step.end - step.start,
    })),
    categories: [...new Set(spans.map((span) => span.category))].sort(
      (a, b) => Object.keys(CATEGORIES).indexOf(a) - Object.keys(CATEGORIES).indexOf(b),
    ),
  };
}

export function timingRunStart(timings) {
  const measured = collect(timings);
  // Avoid spreading one argument per invocation; long runs can exceed the call stack.
  let start = Infinity;
  for (const span of measured) {
    start = Math.min(start, span.start);
  }
  return measured.length ? start : null;
}
