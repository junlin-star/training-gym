const slot = (name) => `var(--color-c-dataviz-${name})`;

// Colour carries the *kind* of cost, not the phase's name: a reader who has
// never seen slime should be able to tell "the GPUs are training" from "we are
// moving weights around" from "nothing is happening" without a glossary.
export const CATEGORIES = {
  train: { label: "Training compute", color: slot("primary-6") },
  generate: { label: "Rollout generation", color: slot("primary-7") },
  reward: { label: "Reward code", color: slot("paired-7") },
  transfer: { label: "Moving weights", color: slot("primary-2") },
  checkpoint: { label: "Checkpointing", color: slot("primary-3") },
  eval: { label: "Eval", color: slot("paired-4") },
  idle: { label: "Waiting / untracked", color: "var(--color-c-gray-30, #6a6a6a)" },
};

// Where two phases of one category sit next to each other often enough that
// telling them apart matters, the second takes a neighbouring tone of the same
// family -- still "this is training", but with a visible seam.
const TONES = {
  compute_log_probs: slot("paired-1"),
  forward_backward: slot("primary-1"),
  optimizer_step: slot("primary-5"),
  offload_train: slot("primary-4"),
  offload_rollout: slot("primary-4"),
};

const PHASE_CATEGORY = {
  train_models: "train",
  compute_log_probs: "train",
  forward_backward: "train",
  optimizer_step: "train",
  generate_rollouts: "generate",
  generate_samples: "generate",
  reward: "reward",
  reward_batch: "reward",
  reward_post_process: "reward",
  weight_sync: "transfer",
  offload_train: "transfer",
  offload_rollout: "transfer",
  checkpoint_save: "checkpoint",
  evaluate_rollouts: "eval",
  evaluate_rollouts_end: "eval",
  wait_for_rollout: "idle",
  wait_for_next_rollout: "idle",
};

export const TIMING_LABELS = {
  evaluate_rollouts: "Eval (before training)",
  evaluate_rollouts_end: "Eval (after training)",
  generate_rollouts: "Waiting for rollouts",
  offload_rollout: "Offload generation engines",
  compute_log_probs: "Log probs",
  train_models: "Train",
  checkpoint_save: "Save checkpoint",
  offload_train: "Offload trainer",
  weight_sync: "Push weights to engines",
  wait_for_rollout: "Waiting for this rollout",
  wait_for_next_rollout: "Waiting for the next rollout",
  generate_samples: "Generate samples",
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
// of intervals: it is drawn as its span with the count and average on it.
const SAMPLED = new Set(["reward", "reward_batch"]);

const NESTS_IN = {
  compute_log_probs: ["train_models"],
  forward_backward: ["train_models"],
  optimizer_step: ["train_models"],
  reward: ["generate_samples"],
  reward_batch: ["generate_samples"],
  reward_post_process: ["generate_samples"],
};

// Two rows, in the order somebody debugging reads them: what the training loop
// itself did, then what the machines generating rollouts did underneath it.
export const GROUPS = [
  {
    key: "step",
    label: "Training loop",
    hint: "the driver and the GPUs it trains on",
    roles: ["driver", "actor", "critic"],
  },
  {
    key: "generation",
    label: "Rollout generation",
    hint: "the inference engines producing samples",
    roles: ["rollout"],
  },
];

const GROUP_OF_ROLE = new Map(
  GROUPS.flatMap((group) => group.roles.map((role) => [role, group.key])),
);

const ROLE_LABELS = {
  driver: "training loop",
  actor: "trainer",
  critic: "critic trainer",
  rollout: "generation engines",
};

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
  return TONES[name] || CATEGORIES[categoryOf(name)].color;
}

export function roleLabel(role) {
  return ROLE_LABELS[role] || role;
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
    const rolloutId = Number(id);
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
          group: GROUP_OF_ROLE.get(role) ?? "step",
          name,
          category: categoryOf(name),
        };
        // Per-sample phases are aggregated even when an older record still
        // carries their runs: thousands of sub-millisecond slivers say less
        // than one span with a count and an average on it.
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
        // only the aggregate, so they are drawn as the span the calls are
        // spread over, carrying their count and average rather than posing as
        // one continuous run.
        spans.push({
          ...where,
          kind: SAMPLED.has(name) || count > 1 ? "sampled" : "work",
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
    if (span.role !== "driver") continue;
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
    if (parent) parent.contains = true;
  }
  return ordered;
}

// One row per worker per depth, split again only when two spans of the same
// worker and depth genuinely overlap -- which in async is the next rollout
// generating during this one's training, and in sync never happens.
function rowsOf(spans, roles) {
  const rows = [];
  const order = (span) => roles.indexOf(span.role);
  for (const span of [...spans].sort(
    (a, b) => order(a) - order(b) || a.depth - b.depth || a.start - b.start,
  )) {
    let row = rows.find(
      (candidate) =>
        candidate.role === span.role &&
        candidate.depth === span.depth &&
        candidate.spans[candidate.spans.length - 1].end <= span.start,
    );
    if (!row) {
      row = { role: span.role, depth: span.depth, spans: [], label: "" };
      rows.push(row);
    }
    row.spans.push(span);
    span.row = rows.indexOf(row);
  }
  // A row is named for the worker it belongs to, and a nested row for the phase
  // its spans ran inside, so depth reads as containment rather than as a gap.
  for (const row of rows) {
    if (!row.depth) {
      row.label = roleLabel(row.role);
      continue;
    }
    const parents = [...new Set(row.spans.map((span) => span.inside))];
    row.label = parents.length === 1 ? `inside ${labelFor(parents[0])}` : "nested";
  }
  return rows;
}

export function runTimeline(timings) {
  const measured = collect(timings);
  if (!measured.length) {
    return { span: 0, runStart: null, groups: [], steps: [], categories: [] };
  }
  const runStart = Math.min(...measured.map((span) => span.start));
  const runEnd = Math.max(...measured.map((span) => span.end));
  const steps = stepsOf(measured);
  const spans = nest([...measured, ...untrackedOf(measured, runStart, runEnd)]);

  for (const span of spans) {
    span.key = `${span.rolloutId}:${span.role}:${span.name}:${span.start.toFixed(3)}`;
    span.offset = span.start - runStart;
    span.duration = span.end - span.start;
    span.average = span.total / span.count;
    span.inside = span.parent ? span.parent.name : null;
    delete span.parent;
  }

  const groups = GROUPS.map((group) => {
    const mine = spans.filter((span) => span.group === group.key);
    return { ...group, rows: rowsOf(mine, group.roles) };
  }).filter((group) => group.rows.length);

  return {
    runStart,
    span: Math.max(runEnd - runStart, 1e-6),
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

export function fmtSecs(s) {
  if (s == null) return "—";
  const n = Number(s);
  if (!Number.isFinite(n)) return "—";
  const trim = (x) => x.toFixed(3).replace(/\.?0+$/, "");
  if (n > 0 && n < 0.01) return `${trim(n * 1000)}ms`;
  if (n >= 60) {
    const m = Math.floor(n / 60);
    return `${m}m ${trim(n - m * 60)}s`;
  }
  return `${trim(n)}s`;
}
