const slot = (name) => `var(--color-c-dataviz-${name})`;

// Colour carries the *kind* of cost, not the phase's name: a reader who has
// never seen slime should be able to tell "the GPUs are training" from "we are
// moving weights around" from "nothing is happening" without a glossary.
export const CATEGORIES = {
  train: { label: "Train", color: slot("primary-7") },
  generate: { label: "Rollout", color: slot("primary-2") },
  transfer: { label: "Weight sync", color: slot("primary-4") },
  checkpoint: { label: "Checkpoint", color: slot("primary-5") },
  eval: { label: "Eval", color: slot("primary-3") },
  idle: { label: "Waiting", color: "var(--color-c-gray-30)" },
};

export const PHASE_CATEGORY = {
  train_models: "train",
  compute_log_probs: "train",
  forward_backward: "train",
  optimizer_step: "train",
  generate_rollouts: "generate",
  generate_samples: "generate",
  sample_generation: "generate",
  reward: "generate",
  reward_batch: "generate",
  reward_post_process: "generate",
  weight_sync: "transfer",
  initial_weight_sync: "transfer",
  offload_train: "transfer",
  offload_rollout: "transfer",
  checkpoint_save: "checkpoint",
  evaluate_rollouts: "eval",
  evaluate_rollouts_end: "eval",
  wait_for_rollout: "idle",
  wait_for_next_rollout: "idle",
};

export const PHASE_COLORS = {
  compute_log_probs:
    "color-mix(in srgb, var(--color-c-dataviz-primary-7) 62%, var(--color-c-gray-02))",
  forward_backward: "var(--color-c-dataviz-training-light)",
  optimizer_step: slot("primary-7"),
  reward:
    "color-mix(in srgb, var(--color-c-dataviz-primary-2) 62%, var(--color-c-gray-02))",
  reward_batch:
    "color-mix(in srgb, var(--color-c-dataviz-primary-2) 62%, var(--color-c-gray-02))",
  reward_post_process:
    "color-mix(in srgb, var(--color-c-dataviz-primary-2) 62%, var(--color-c-gray-02))",
  sample_generation: slot("primary-2"),
};

export const TIMING_LABELS = {
  evaluate_rollouts: "Eval (before training)",
  evaluate_rollouts_end: "Eval (after training)",
  generate_rollouts: "Waiting for rollouts",
  offload_rollout: "Offload generation engines",
  compute_log_probs: "Calculate log probs",
  train_models: "Train",
  checkpoint_save: "Save checkpoint",
  offload_train: "Offload trainer",
  weight_sync: "Push weights to engines",
  initial_weight_sync: "Initial weight sync",
  wait_for_rollout: "Waiting for this rollout",
  wait_for_next_rollout: "Waiting for the next rollout",
  generate_samples: "Rollout generation",
  sample_generation: "Sample generation",
  reward: "Reward",
  reward_batch: "Reward (whole batch)",
  reward_post_process: "Reward post process",
  forward_backward: "Forward/backward",
  optimizer_step: "Optimizer step",
  untracked: "Untracked",
};

// Phases where the worker whose lane they are on is blocked on somebody else:
// they are drawn as stalls, and the work itself shows up on the row of the
// worker actually doing it.
const STALLS = new Set([
  "generate_rollouts",
  "wait_for_rollout",
  "wait_for_next_rollout",
  "evaluate_rollouts",
  "evaluate_rollouts_end",
]);

// A phase measured once per sample, kept as an aggregate rather than thousands
// of intervals.
const SAMPLED = new Set(["reward", "reward_batch", "sample_generation"]);
export const HIDDEN_PHASES = new Set(["reward", "reward_batch", "reward_post_process"]);

const NESTS_IN = {
  compute_log_probs: ["train_models"],
  forward_backward: ["train_models"],
  optimizer_step: ["train_models"],
  reward: ["generate_samples"],
  reward_batch: ["generate_samples"],
  reward_post_process: ["generate_samples"],
  sample_generation: ["generate_samples"],
};

export const GROUPS = [
  {
    key: "timeline",
    label: "Timeline",
    hint: "measured phases on the shared wall clock",
  },
];

const NEGLIGIBLE_WORK_S = 0.0005;
// Below this a hole is loop overhead rather than something to go and look at.
const UNTRACKED_FLOOR_S = 0.25;

export function labelFor(name) {
  return TIMING_LABELS[name] || name.replace(/_/g, " ");
}

export function categoryOf(name) {
  return PHASE_CATEGORY[name] || "idle";
}

export function colorFor(name) {
  return PHASE_COLORS[name] || CATEGORIES[categoryOf(name)].color;
}

function merge(spans) {
  const merged = [];
  for (const [start, end] of [...spans].sort((a, b) => a[0] - b[0])) {
    const last = merged[merged.length - 1];
    if (last && start <= last[1]) last[1] = Math.max(last[1], end);
    else merged.push([start, end]);
  }
  return merged;
}

function collect(timings) {
  const spans = [];
  for (const [id, lanes] of Object.entries(timings || {})) {
    const parsedId = Number(id);
    const rolloutId =
      id === "bootstrap"
        ? null
        : Number.isFinite(parsedId) && Number.isInteger(parsedId)
          ? parsedId
          : null;
    for (const [role, lane] of Object.entries(lanes?.roles || {})) {
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
        // Per-sample phases are aggregated even when an older record still
        // carries their runs: thousands of sub-millisecond slivers say less
        // than one span across the calls' spread.
        const runs =
          !SAMPLED.has(name) && Array.isArray(phase?.invocations)
            ? phase.invocations
            : [];
        // A phase that ran more than once is drawn as the runs themselves. Its
        // first start to its last end is not an interval it was ever inside:
        // an async step's two waits straddle the training between them.
        if (runs.length) {
          for (const [from, to] of runs) {
            const start = laneStart + (Number(from) || 0);
            const end = laneStart + (Number(to) || 0);
            spans.push({
              ...where,
              kind: STALLS.has(name) ? "stall" : "work",
              start,
              end,
              count: 1,
              total: end - start,
              longest: end - start,
            });
          }
          continue;
        }
        // Per-sample phases (and records written before runs were kept) have
        // only aggregate timing, so they are drawn across the calls' spread.
        spans.push({
          ...where,
          kind: STALLS.has(name)
            ? "stall"
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
  return spans;
}

// Where each rollout's step sits on the run's clock, taken from the lane the
// loop itself runs on so a step spans exactly what the driver did for it.
function stepsOf(spans) {
  const byRollout = new Map();
  for (const span of spans) {
    if (span.role !== "driver" || span.rolloutId == null) continue;
    const step = byRollout.get(span.rolloutId) ?? {
      id: span.rolloutId,
      start: span.start,
      end: span.end,
      work: 0,
      stalled: 0,
    };
    step.start = Math.min(step.start, span.start);
    step.end = Math.max(step.end, span.end);
    if (span.kind === "stall") step.stalled += span.total;
    else step.work += span.total;
    byRollout.set(span.rolloutId, step);
  }
  return [...byRollout.values()].sort((a, b) => a.start - b.start);
}

// Wall-clock the loop spent somewhere no phase measured. Drawn rather than left
// blank: a hole in the driver's timeline is a finding, not decoration.
function untrackedOf(spans, runStart, runEnd) {
  const driver = spans.filter((span) => span.role === "driver");
  if (!driver.length) return [];
  const covered = merge(driver.map((span) => [span.start, span.end]));
  const holes = [];
  let cursor = runStart;
  for (const [start, end] of covered) {
    if (start - cursor >= UNTRACKED_FLOOR_S) holes.push([cursor, start]);
    cursor = Math.max(cursor, end);
  }
  if (runEnd - cursor >= UNTRACKED_FLOOR_S) holes.push([cursor, runEnd]);
  return holes.map(([start, end]) => ({
    rolloutId: null,
    role: "driver",
    group: "step",
    name: "untracked",
    category: "idle",
    kind: "untracked",
    start,
    end,
    count: 1,
    total: end - start,
    longest: end - start,
  }));
}

function nest(spans) {
  const ordered = [...spans].sort((a, b) => a.start - b.start || b.end - a.end);
  for (const span of ordered) {
    const parent = ordered
      .filter(
        (other) =>
          other !== span &&
          other.rolloutId === span.rolloutId &&
          other.group === span.group &&
          other.start <= span.start &&
          span.end <= other.end &&
          (NESTS_IN[span.name] || []).includes(other.name),
      )
      .sort((a, b) => b.depth - a.depth)[0];
    span.depth = parent ? parent.depth + 1 : 0;
    span.parent = parent ?? null;
    span.children = [];
    if (parent) parent.contains = true;
    if (parent) parent.children.push(span);
  }
  for (const span of ordered) {
    const occurrences = new Map();
    for (const child of [...span.children].sort((a, b) => a.start - b.start)) {
      const occurrence = (occurrences.get(child.name) || 0) + 1;
      occurrences.set(child.name, occurrence);
      child.ordinal = occurrence;
    }
    const duration = Math.max(span.end - span.start, 0);
    const children = new Map();
    for (const child of span.children) {
      const childDuration = Math.max(child.end - child.start, 0);
      const current = children.get(child.name) ?? {
        name: child.name,
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
      children.set(child.name, current);
    }
    span.children = [...children.values()].map((child) => ({
      ...child,
      share: duration > 0 ? child.duration / duration : 0,
      average: child.count ? child.total / child.count : 0,
    }));
  }
  return ordered;
}

function clipStalls(spans, async) {
  const workByRow = new Map();
  for (const span of spans) {
    if (span.kind === "stall" || span.kind === "untracked") continue;
    const row = async && span.role === "rollout" ? "generation" : "step";
    const intervals = workByRow.get(row) || [];
    intervals.push([span.start, span.end]);
    workByRow.set(row, intervals);
  }
  const clipped = [];
  for (const span of spans) {
    if (span.kind !== "stall") {
      clipped.push(span);
      continue;
    }
    let pieces = [[span.start, span.end]];
    for (const [workStart, workEnd] of workByRow.get(
      async && span.role === "rollout" ? "generation" : "step",
    ) || []) {
      pieces = pieces.flatMap(([start, end]) =>
        end <= workStart || start >= workEnd
          ? [[start, end]]
          : [
              ...(start < workStart ? [[start, workStart]] : []),
              ...(end > workEnd ? [[workEnd, end]] : []),
            ],
      );
    }
    for (const [start, end] of pieces) {
      if (end - start >= NEGLIGIBLE_WORK_S) {
        clipped.push({
          ...span,
          start,
          end,
          total: end - start,
          duration: end - start,
        });
      }
    }
  }
  return clipped;
}

function rowsOf(spans, async) {
  const driverSpans = spans.filter((span) => !async || span.role !== "rollout");
  if (!async) {
    return [
      {
        key: "driver",
        label: "Training loop",
        hint: "Driver and trainer phases on the shared wall clock.",
        spans: driverSpans,
      },
    ];
  }

  const rows = [
    {
      key: "driver",
      label: "Driver",
      hint: "Driver and trainer phases on the shared wall clock.",
      spans: driverSpans,
    },
  ];
  const rolloutSpans = spans.filter((span) => span.role === "rollout");
  const roots = rolloutSpans.filter((span) => span.depth === 0);
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
    const ids = [
      ...new Set(
        packedRow.roots
          .map((span) => span.rolloutId)
          .filter((id) => id != null),
      ),
    ].sort((a, b) => a - b);
    rows.push({
      key: `rollout-${index}`,
      label: "ROLLOUTS",
      hint: "Rollout engine phases packed by their actual wall-clock overlap.",
      spans: rolloutSpans.filter((span) => {
        let root = span;
        while (root.parent) root = root.parent;
        return rootSet.has(root);
      }),
    });
  }
  return rows;
}

export function runTimeline(timings) {
  const measured = collect(timings);
  if (!measured.length) {
    return { span: 0, runStart: null, async: false, groups: [], steps: [], categories: [] };
  }
  const runStart = Math.min(...measured.map((span) => span.start));
  const runEnd = Math.max(...measured.map((span) => span.end));
  const steps = stepsOf(measured);
  const rawSpans = [...measured, ...untrackedOf(measured, runStart, runEnd)];
  const generationSpans = rawSpans.filter((span) => span.role === "rollout");
  const stepSpans = rawSpans.filter(
    (span) =>
      span.role !== "rollout" &&
      span.kind !== "stall" &&
      span.kind !== "untracked",
  );
  const async = generationSpans.some((generation) =>
    stepSpans.some(
      (step) => generation.start < step.end && step.start < generation.end,
    ),
  );
  const spans = nest(clipStalls(rawSpans, async));

  for (const span of spans) {
    span.key = `${span.rolloutId}:${span.role}:${span.name}:${span.start.toFixed(3)}`;
  }
  for (const span of spans) {
    span.offset = span.start - runStart;
    span.duration = span.end - span.start;
    span.average = span.total / span.count;
    span.inside = span.parent ? span.parent.name : null;
    span.insideKey = span.parent ? span.parent.key : null;
  }

  const rows = rowsOf(spans, async);
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
    untracked: spans
      .filter((span) => span.kind === "untracked")
      .reduce((total, span) => total + span.duration, 0),
    categories: [...new Set(spans.map((span) => span.category))].sort(
      (a, b) => Object.keys(CATEGORIES).indexOf(a) - Object.keys(CATEGORIES).indexOf(b),
    ),
  };
}

export function fmtSecs(s, unit = null) {
  if (s == null) return "—";
  const n = Number(s);
  if (!Number.isFinite(n)) return "—";
  const trim = (x) => x.toFixed(3).replace(/\.?0+$/, "");
  if (unit === "ms") return `${trim(n * 1000)}ms`;
  if (unit === "s") return `${trim(n)}s`;
  if (n > 0 && n < 0.01) return `${trim(n * 1000)}ms`;
  if (n >= 60) {
    const m = Math.floor(n / 60);
    return `${m}m ${trim(n - m * 60)}s`;
  }
  return `${trim(n)}s`;
}
