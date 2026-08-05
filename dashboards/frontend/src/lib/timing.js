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

/** Every measured phase run of one rollout, on the time axis its lanes share.
 *
 * `lanes` is one rollout's `{roles: {role: lane}}` from the timings API. Lanes
 * are recorded in different processes, each phase offset relative to its own
 * lane's start, so `lane_start_unix_s` shifts them onto one axis.
 *
 * A phase that recorded its runs contributes one bar per run -- that is what
 * makes `forward_backward` and `optimizer_step` read as alternating rather than
 * one containing the other. A phase that ran too many times to keep them
 * (rewards, one run per sample) contributes a single band over the span it
 * covered, carrying `count`, `average` and `longest` instead.
 *
 * Rows follow what the times mean. A bar that ran entirely inside another --
 * `forward_backward` inside `train_models` -- is nested one level under it, so
 * the sync path reads as one timeline of thin bars inside their step. A bar
 * gets a row of its own only when it overlaps something it is not inside, which
 * is real concurrency: an async rollout generating while the actor trains. Each
 * bar carries the relationship it was drawn with -- `inside` and `overlaps` --
 * so hovering says why it sits where it does rather than leaving the reader to
 * guess whether an overlap is intended.
 *
 * A band is not a container: it covers the span its runs were scattered over
 * rather than one continuous run, so work drawn beneath it is not part of it.
 */
export function rolloutTimeline(lanes) {
  const roles = Object.entries(lanes?.roles || {});
  const laneStarts = roles
    .map(([, lane]) => Number(lane?.lane_start_unix_s))
    .filter((s) => Number.isFinite(s));
  const earliestLaneStart = laneStarts.length ? Math.min(...laneStarts) : null;

  const bars = [];
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
      if (runs.length) {
        runs.forEach(([start, end], i) => {
          bars.push({
            key: `${role}:${name}:${i}`,
            role,
            name,
            start: Number(start) + shift,
            end: Number(end) + shift,
            duration: Number(end) - Number(start),
            count: 1,
            banded: false,
          });
        });
      } else if (count) {
        bars.push({
          key: `${role}:${name}`,
          role,
          name,
          start: (Number(phase?.first_start_s) || 0) + shift,
          end: (Number(phase?.last_end_s) || 0) + shift,
          duration: total,
          count,
          average: total / count,
          longest: Number(phase?.longest_duration_s) || 0,
          // A phase that ran once spans exactly its one run, so the band is
          // that run; anything else covers runs the record no longer holds.
          banded: count > 1,
        });
      }
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
    if (!bar.banded) enclosing.push(bar);
  }

  // Work that ran at the same time without either side containing the other:
  // the only shape that should read as concurrency.
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

  // One row per nesting depth, and another at that depth for a bar that
  // overlaps a bar it is not inside.
  const rows = [];
  for (const bar of bars) {
    const row = rows.find(
      (r) => r[0].depth === bar.depth && r[r.length - 1].end <= bar.start,
    );
    if (row) row.push(bar);
    else rows.push([bar]);
  }
  rows.sort((a, b) => a[0].depth - b[0].depth || a[0].start - b[0].start);

  const span = bars.length ? Math.max(...bars.map((bar) => bar.end)) : 0;
  return { rows, span };
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
