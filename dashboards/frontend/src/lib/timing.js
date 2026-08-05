export const TIMING_LABELS = {
  evaluate_rollouts: "Eval (before)",
  generate_rollouts: "Generate rollouts",
  offload_rollout: "Offload rollout",
  compute_log_probs: "Compute log probs",
  train_models: "Train models",
  checkpoint_save: "Checkpoint save",
  offload_train: "Offload train",
  weight_sync: "Weight sync",
  evaluate_rollouts_end: "Eval (after)",
  wait_for_rollout: "Wait for rollout",
  generate_samples: "Generate samples",
  reward: "Reward",
  reward_batch: "Reward (batch)",
  reward_post_process: "Reward post process",
  forward_backward: "Forward/backward",
  optimizer_step: "Optimizer step",
};

// Dataviz slots, assigned along the step rather than by family: a phase and the
// one it runs next to are neighbouring slots (mint → teal → blue through
// generation and training, warm only for the work beside the step), so nothing
// clashes edge to edge and no phase is told apart by a shade of another.
export const TIMING_COLORS = {
  generate_rollouts: "#ADEAAB",
  generate_samples: "#4AA19D",
  reward: "#C4687F",
  reward_batch: "#B0566C",
  reward_post_process: "#8D324C",
  wait_for_rollout: "#DECB6C",
  offload_rollout: "#4AA19D",
  train_models: "#648FE0",
  compute_log_probs: "#648FE0",
  forward_backward: "#FFC1F7",
  optimizer_step: "#DECB6C",
  offload_train: "#4AA19D",
  weight_sync: "#8D324C",
  checkpoint_save: "#D9866B",
  evaluate_rollouts: "#FFC1F7",
  evaluate_rollouts_end: "#D9866B",
};

// Work a step waits on but isn't: a checkpoint or an eval lands on one rollout
// and would make that step read as many times slower than its peers.
export const PHASES_BESIDE_THE_STEP = [
  "checkpoint_save",
  "evaluate_rollouts",
  "evaluate_rollouts_end",
];

// Below this a phase did no measurable work, so it is recorded but not drawn.
const NEGLIGIBLE_WORK_S = 0.0005;

// What a phase is drawn inside when the two ran on different lanes: the phase
// that blocked on it, so the driver's train_models really is the actor's
// forward/backward. Anything else measured on another lane merely ran while the
// phase its times fall in was running — a rollout worker generating during
// training — and takes a row of its own.
const BLOCKED_ON = {
  compute_log_probs: ["train_models"],
  forward_backward: ["train_models"],
  optimizer_step: ["train_models"],
  generate_samples: ["generate_rollouts"],
  reward: ["generate_samples", "generate_rollouts"],
  reward_batch: ["generate_samples", "generate_rollouts"],
  reward_post_process: ["generate_samples", "generate_rollouts"],
};

function nestsWithin(bar, container) {
  return (
    bar.role === container.role ||
    (BLOCKED_ON[bar.name] || []).includes(container.name)
  );
}

export function labelFor(name) {
  return TIMING_LABELS[name] || name.replace(/_/g, " ");
}

export function colorFor(name) {
  return TIMING_COLORS[name] || "var(--color-c-gray-40, #5e5e5e)";
}

/** Every measured phase run of one rollout, on the time axis its lanes share.
 *
 * `lanes` is one rollout's `{roles: {role: lane}}` from the timings API. Each
 * lane's offsets are relative to its own start, so `lane_start_unix_s` shifts
 * them onto one axis.
 *
 * A phase contributes one bar per recorded run. A phase measured once per sample
 * keeps no runs to draw, so it reads as work spent inside the bar that contains
 * it (`spent`) rather than as a block of its own, which would be mostly the gaps
 * between its calls. A run entirely inside another is nested within it; a bar
 * takes a row of its own only where it overlaps work it is not inside, which is
 * real concurrency.
 */
export function rolloutTimeline(lanes) {
  const roles = Object.entries(lanes?.roles || {});
  const laneStarts = roles
    .map(([, lane]) => Number(lane?.lane_start_unix_s))
    .filter((s) => Number.isFinite(s));
  const earliestLaneStart = laneStarts.length ? Math.min(...laneStarts) : null;

  const bars = [];
  const perSample = [];
  for (const [role, lane] of roles) {
    const laneStart = Number(lane?.lane_start_unix_s);
    const shift =
      earliestLaneStart != null && Number.isFinite(laneStart)
        ? laneStart - earliestLaneStart
        : 0;
    for (const [name, phase] of Object.entries(lane?.phases || {})) {
      const count = Number(phase?.count) || 0;
      const total = Number(phase?.total_duration_s) || 0;
      const runs = Array.isArray(phase?.invocations) ? phase.invocations : [];
      const first = (Number(phase?.first_start_s) || 0) + shift;
      const last = (Number(phase?.last_end_s) || 0) + shift;
      if (!count || total < NEGLIGIBLE_WORK_S) continue;
      // A phase scored per sample reads as work spent inside the phase it ran
      // in: its runs are too brief to draw, and past the recorder's cap it
      // keeps none of them.
      if (total / count < NEGLIGIBLE_WORK_S || (!runs.length && count > 1)) {
        perSample.push({
          role,
          name,
          count,
          total,
          longest: Number(phase?.longest_duration_s) || 0,
          start: first,
          end: last,
        });
        continue;
      }
      // A phase that ran once spans exactly its one run, so a pre-cutover
      // record still draws as the run it measured.
      const drawn = runs.length ? runs : [[first - shift, last - shift]];
      // Runs of one phase that overlap are one block of work.
      const blocks = [];
      for (const [start, end] of [...drawn].sort((a, b) => a[0] - b[0])) {
        const block = blocks[blocks.length - 1];
        if (block && Number(start) <= block.end) {
          block.end = Math.max(block.end, Number(end));
          block.runs += 1;
          block.work += Number(end) - Number(start);
        } else {
          blocks.push({
            start: Number(start),
            end: Number(end),
            runs: 1,
            work: Number(end) - Number(start),
          });
        }
      }
      blocks.forEach((block, index) => {
        bars.push({
          key: `${role}:${name}:${index}`,
          role,
          name,
          start: block.start + shift,
          end: block.end + shift,
          duration: block.end - block.start,
          runs: block.runs,
          work: block.work,
          contains: false,
          spent: [],
        });
      });
    }
  }
  // Enclosing bars first, so a bar meets its container before itself.
  bars.sort((a, b) => a.start - b.start || b.end - a.end);

  bars.forEach((bar, index) => {
    // The innermost phase this one both ran within and belongs to.
    const container = bars
      .slice(0, index)
      .filter(
        (other) =>
          other.start <= bar.start &&
          bar.end <= other.end &&
          nestsWithin(bar, other),
      )
      .sort((a, b) => b.depth - a.depth)[0];
    bar.depth = container ? container.depth + 1 : 0;
    bar.container = container ?? null;
    bar.inside = container ? container.name : null;
    // A bar with work inside it is drawn as an outline around that work.
    if (container) container.contains = true;
  });

  // A per-sample phase's work reads on the innermost bar it ran within
  // ("generation spent 32ms scoring 227 samples"); one nothing contains keeps a
  // bar of its own rather than going unshown.
  for (const work of perSample) {
    const container = bars
      .filter(
        (bar) =>
          bar.start <= work.start &&
          work.end <= bar.end &&
          nestsWithin(work, bar),
      )
      .sort((a, b) => b.depth - a.depth)[0];
    if (container) container.spent.push(work);
    else
      bars.push({
        key: `${work.role}:${work.name}`,
        ...work,
        duration: work.total,
        runs: work.count,
        work: work.total,
        depth: 0,
        container: null,
        inside: null,
        contains: false,
        spent: [work],
      });
  }

  // Work that ran at the same time without either side containing the other.
  for (const bar of bars) {
    bar.overlaps = [
      ...new Set(
        bars
          .filter(
            (other) =>
              other !== bar &&
              other.start < bar.end &&
              bar.start < other.end &&
              !(other.start <= bar.start && bar.end <= other.end) &&
              !(bar.start <= other.start && other.end <= bar.end),
          )
          .map((other) => other.name),
      ),
    ];
  }

  // A row holds bars that ran one after another, and belongs to the row holding
  // the bars that contain them, so it is drawn within that row and never within
  // a bar it is not inside. Work that overlaps its own row takes another one.
  bars.sort((a, b) => a.depth - b.depth || a.start - b.start || b.end - a.end);
  const rows = [];
  const rowOf = new Map();
  const rowsPerParent = new Map();
  for (const bar of bars) {
    const parent = rowOf.get(bar.container) ?? null;
    const siblings = rowsPerParent.get(parent) || [];
    let row = siblings.find((r) => r.bars[r.bars.length - 1].end <= bar.start);
    if (!row) {
      // The first row of a container is drawn within it; another one is work
      // that overlapped it, which needs a row of its own.
      row = {
        depth: bar.depth,
        parentIndex: parent ? parent.index : null,
        concurrent: siblings.length > 0,
        bars: [],
      };
      row.index = rows.push(row) - 1;
      rowsPerParent.set(parent, [...siblings, row]);
    }
    row.bars.push(bar);
    rowOf.set(bar, row);
  }

  const span = bars.length ? Math.max(...bars.map((bar) => bar.end)) : 0;
  const beside = bars.filter((bar) =>
    PHASES_BESIDE_THE_STEP.includes(bar.name),
  );
  // The driver runs its substeps one after another, so a step is their work
  // added up; a checkpoint or an eval is not one of them.
  const stepDuration = bars
    .filter(
      (bar) =>
        bar.role === "driver" && !PHASES_BESIDE_THE_STEP.includes(bar.name),
    )
    .reduce((total, bar) => total + bar.work, 0);
  return {
    rows,
    span,
    stepDuration,
    beside: beside.map((bar) => ({ name: bar.name, duration: bar.duration })),
  };
}

export function fmtSecs(s) {
  if (s == null) return "—";
  const n = Number(s);
  if (!Number.isFinite(n)) return "—";
  const trim = (x) => x.toFixed(3).replace(/\.?0+$/, "");
  // A per-sample phase averages well under a millisecond; seconds would read 0s.
  if (n > 0 && n < 0.01) return `${trim(n * 1000)}ms`;
  if (n >= 60) {
    const m = Math.floor(n / 60);
    return `${m}m ${trim(n - m * 60)}s`;
  }
  return `${trim(n)}s`;
}
