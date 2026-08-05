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
  custom_reward: "Custom reward",
  reward_post_process: "Reward post process",
  forward_backward: "Forward/backward",
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
  custom_reward: "#a3e635",
  reward_post_process: "#f0abfc",
  forward_backward: "#2dd4bf",
};

export function labelFor(name) {
  return TIMING_LABELS[name] || name.replace(/_/g, " ");
}

export function colorFor(name) {
  return TIMING_COLORS[name] || "var(--color-c-gray-40, #5e5e5e)";
}

export function phaseSummaries(lanes = {}) {
  const roles = lanes.roles || lanes || {};
  const rolloutIds = new Set();
  const totalsByName = {};

  for (const [role, lane] of Object.entries(roles)) {
    const phaseTotals = lane?.totals || {};
    const laneRolloutId = lane?.role === role ? lane?.rollout_id : undefined;
    if (laneRolloutId !== undefined) rolloutIds.add(laneRolloutId);
    for (const [name, t] of Object.entries(phaseTotals)) {
      if (!totalsByName[name]) {
        totalsByName[name] = {
          name,
          count: 0,
          total_duration_s: 0,
          max_duration_s: 0,
          rollouts: new Set(),
        };
      }
      const entry = totalsByName[name];
      const c = Number(t?.count) || 0;
      const total = Number(t?.total_duration_s) || 0;
      const max = Number(t?.max_duration_s) || 0;
      entry.count += c;
      entry.total_duration_s += total;
      entry.max_duration_s = Math.max(entry.max_duration_s, max);
      entry.rollouts.add(role);
    }
  }

  const rolloutCount = rolloutIds.size || Object.keys(roles).length || 1;
  return Object.values(totalsByName).map((row) => ({
    name: row.name,
    count: row.rollouts.size,
    totalDuration: row.total_duration_s,
    avgDuration: row.count ? row.total_duration_s / row.count : 0,
    maxDuration: row.max_duration_s,
    rolloutCount,
  }));
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
