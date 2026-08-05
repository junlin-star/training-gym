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

const slot = (name) => `var(--color-c-dataviz-${name})`;

export const TIMING_COLORS = {
  generate_rollouts: slot("primary-1"),
  generate_samples: slot("primary-6"),
  wait_for_rollout: slot("paired-1"),
  reward: slot("primary-3"),
  reward_batch: slot("paired-3"),
  reward_post_process: slot("paired-7"),
  train_models: slot("primary-7"),
  compute_log_probs: slot("paired-4"),
  forward_backward: slot("primary-4"),
  optimizer_step: `color-mix(in srgb, ${slot("primary-7")} 55%, white)`,
  weight_sync: slot("paired-8"),
  offload_rollout: slot("paired-5"),
  offload_train: slot("primary-2"),
  checkpoint_save: slot("paired-2"),
  evaluate_rollouts: slot("primary-5"),
  evaluate_rollouts_end: slot("paired-6"),
};

export const PHASES_BESIDE_THE_STEP = [
  "checkpoint_save",
  "evaluate_rollouts",
  "evaluate_rollouts_end",
];

const NEGLIGIBLE_WORK_S = 0.0005;

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
      const drawn = runs.length ? runs : [[first - shift, last - shift]];
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
          band: null,
          spent: [],
        });
      });
    }
  }
  bars.sort((a, b) => a.start - b.start || b.end - a.end);

  bars.forEach((bar, index) => {
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
    if (container) container.contains = true;
  });

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
        runs: 1,
        work: work.total,
        depth: 0,
        container: null,
        inside: null,
        contains: false,
        band: work,
        spent: [],
      });
  }

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

  bars.sort((a, b) => a.depth - b.depth || a.start - b.start || b.end - a.end);
  const rows = [];
  const rowOf = new Map();
  const rowsPerParent = new Map();
  for (const bar of bars) {
    const parent = rowOf.get(bar.container) ?? null;
    const siblings = rowsPerParent.get(parent) || [];
    let row = siblings.find((r) => r.bars[r.bars.length - 1].end <= bar.start);
    if (!row) {
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
  if (n > 0 && n < 0.01) return `${trim(n * 1000)}ms`;
  if (n >= 60) {
    const m = Math.floor(n / 60);
    return `${m}m ${trim(n - m * 60)}s`;
  }
  return `${trim(n)}s`;
}
