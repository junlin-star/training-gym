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

// A phase is coloured by the work it belongs to -- generation green, training
// blue, moving weights or samples around pink, and the rest of a step grey --
// so a rollout reads as a few pieces of work rather than a row of unrelated bars.
export const TIMING_COLORS = {
  generate_rollouts: "var(--color-c-green-70)",
  generate_samples: "var(--color-c-pale-green-90)",
  reward: "var(--color-c-pale-green-60)",
  reward_batch: "var(--color-c-pale-green-50)",
  reward_post_process: "var(--color-c-pale-green-40)",
  train_models: "var(--color-c-blue-60)",
  compute_log_probs: "var(--color-c-blue-90)",
  forward_backward: "var(--color-c-blue-100)",
  optimizer_step: "var(--color-c-blue-40)",
  offload_rollout: "var(--color-c-pink-50)",
  offload_train: "var(--color-c-pink-40)",
  weight_sync: "var(--color-c-pink-80)",
  checkpoint_save: "var(--color-c-yellow-70)",
  evaluate_rollouts: "var(--color-c-gray-60)",
  evaluate_rollouts_end: "var(--color-c-gray-50)",
  wait_for_rollout: "var(--color-c-gray-30)",
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

  const enclosing = [];
  for (const bar of bars) {
    while (enclosing.length && enclosing[enclosing.length - 1].end < bar.end) {
      enclosing.pop();
    }
    const container = enclosing[enclosing.length - 1];
    bar.depth = enclosing.length;
    bar.inside = container ? container.name : null;
    // A bar with work inside it is drawn as an outline around that work.
    if (container) container.contains = true;
    enclosing.push(bar);
  }

  // A per-sample phase's work reads on the innermost bar it ran within
  // ("generation spent 32ms scoring 227 samples"); one nothing contains keeps a
  // bar of its own rather than going unshown.
  for (const work of perSample) {
    const container = bars
      .filter((bar) => bar.start <= work.start && work.end <= bar.end)
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

  // One row per nesting depth, drawn within the row that contains it, and
  // another at that depth for a bar that overlaps a bar it is not inside.
  bars.sort((a, b) => a.start - b.start || b.end - a.end);
  const rows = [];
  for (const bar of bars) {
    const row = rows.find(
      (r) =>
        r.depth === bar.depth && r.bars[r.bars.length - 1].end <= bar.start,
    );
    if (row) row.bars.push(bar);
    else rows.push({ depth: bar.depth, bars: [bar] });
  }
  rows.sort((a, b) => a.depth - b.depth || a.bars[0].start - b.bars[0].start);
  const drawnDepths = new Set();
  for (const row of rows) {
    row.concurrent = drawnDepths.has(row.depth);
    drawnDepths.add(row.depth);
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
