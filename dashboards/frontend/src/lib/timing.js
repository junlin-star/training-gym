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
  reward_post_process: "Reward post process",
  forward_backward: "Forward/backward",
  optimizer_step: "Optimizer step",
};

export const TIMING_COLORS = {
  evaluate_rollouts: "#60a5fa",
  generate_rollouts: "#34d399",
  offload_rollout: "#a78bfa",
  compute_log_probs: "#fbbf24",
  train_models: "#f87171",
  checkpoint_save: "#f472b6",
  offload_train: "#c084fc",
  weight_sync: "#22d3ee",
  evaluate_rollouts_end: "#818cf8",
  wait_for_rollout: "#fb923c",
  generate_samples: "#4ade80",
  reward: "#a3e635",
  reward_post_process: "#f0abfc",
  forward_backward: "#2dd4bf",
  optimizer_step: "#facc15",
};

export function labelFor(name) {
  return TIMING_LABELS[name] || name.replace(/_/g, " ");
}

export function colorFor(name) {
  return TIMING_COLORS[name] || "var(--color-c-gray-40, #5e5e5e)";
}

/** Per-phase totals across the rollouts currently listed, longest first.
 *
 * `timings` is the `{rollout_id: {roles: {role: lane}}}` map the timings API
 * returns, so `rolloutsMeasured` counts the rollouts that recorded a phase and
 * `rolloutCount` how many were asked for: fewer means some rollout's lane is
 * missing, which the summary says out loud rather than averaging over.
 */
export function phaseSummaries(timings = {}) {
  const rollouts = Object.values(timings);
  const byName = {};

  for (const lanes of rollouts) {
    const seenHere = new Set();
    for (const lane of Object.values(lanes?.roles || {})) {
      for (const [name, phase] of Object.entries(lane?.phases || {})) {
        const row = (byName[name] ??= {
          name,
          count: 0,
          totalDuration: 0,
          longestDuration: 0,
          rolloutsMeasured: 0,
        });
        row.count += Number(phase?.count) || 0;
        row.totalDuration += Number(phase?.total_duration_s) || 0;
        row.longestDuration = Math.max(
          row.longestDuration,
          Number(phase?.longest_duration_s) || 0,
        );
        if (!seenHere.has(name)) {
          seenHere.add(name);
          row.rolloutsMeasured += 1;
        }
      }
    }
  }

  const rolloutCount = rollouts.length;
  return Object.values(byName)
    .map((row) => ({
      ...row,
      avgDuration: row.count ? row.totalDuration / row.count : 0,
      avgPerRollout: row.rolloutsMeasured
        ? row.totalDuration / row.rolloutsMeasured
        : 0,
      rolloutCount,
    }))
    .sort((a, b) => b.totalDuration - a.totalDuration);
}

export function fmtSecs(s) {
  if (s == null) return "—";
  const n = Number(s);
  if (!Number.isFinite(n)) return "—";
  const trim = (x) => x.toFixed(3).replace(/\.?0+$/, "");
  if (n >= 60) {
    const m = Math.floor(n / 60);
    return `${m}m ${trim(n - m * 60)}s`;
  }
  return `${trim(n)}s`;
}
