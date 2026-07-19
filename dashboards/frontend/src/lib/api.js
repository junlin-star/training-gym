const SERVER = "/api";

function safeStr(v) {
  if (v && typeof v === "object" && "value" in v) return v.value;
  return v != null ? String(v) : "";
}

function safeTimestamp(v) {
  if (v && typeof v === "object" && "value" in v) return safeTimestamp(v.value);
  if (typeof v === "number") return Number.isFinite(v) ? v : 0;
  if (typeof v === "string" && v.trim()) {
    const parsed = Number(v);
    if (Number.isFinite(parsed)) return parsed;
    const epochMs = Date.parse(v);
    if (Number.isFinite(epochMs)) return Math.floor(epochMs / 1000);
  }
  return 0;
}

function modalAppUrl(modalAppId) {
  const appId = safeStr(modalAppId).trim();
  if (!appId) return null;
  if (appId.startsWith("http://") || appId.startsWith("https://")) return appId;
  return `https://modal.com/id/${appId}`;
}

function summarizeDeployment(d) {
  const config = d.deployment_config || {};
  const model = config.model || {};
  const checkpoint = config.checkpoint || {};
  const health = d.health || {};
  const appId = safeStr(d.modal_app_id || "");
  const rawStatus =
    safeStr(d.status || "") ||
    safeStr(d.deployment_status || "") ||
    safeStr(d.state || "") ||
    safeStr(d.health_status || "") ||
    safeStr(health.status || "") ||
    safeStr(config.status || "") ||
    safeStr(config.state || "");
  return {
    deployment_id: safeStr(d.deployment_id || ""),
    app_name: safeStr(config.app_name || ""),
    served_model_name: safeStr(config.served_model_name || ""),
    model_name: safeStr(model.model_name || ""),
    model_path: safeStr(model.model_path || ""),
    checkpoint_path: safeStr(checkpoint.path || ""),
    status: rawStatus,
    version: safeStr(d.version || d.deployment_version || ""),
    created_at: safeTimestamp(d.created_at || d.updated_at || d.last_updated_at),
    url: safeStr(d.url || ""),
    modal_app_id: appId,
    modal_app_url: modalAppUrl(appId),
  };
}

export async function fetchRuns({ signal } = {}) {
  const response = await fetch(`${SERVER}/runs`, { signal });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const runs = await response.json();
  return Array.isArray(runs) ? runs : [];
}

export async function fetchEvals({ signal } = {}) {
  const res = await fetch(`${SERVER}/evals`, { signal });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const evals = await res.json();
  const seen = new Set();
  return evals.filter((e) => {
    if (seen.has(e.eval_id)) return false;
    seen.add(e.eval_id);
    return true;
  });
}

export async function fetchDeployments({ signal } = {}) {
  const res = await fetch(`${SERVER}/deployments`, { signal });
  if (!res.ok) return [];
  const deployments = await res.json();
  return deployments.map(summarizeDeployment);
}

export async function fetchTrainResult(trainingRunId) {
  const res = await fetch(
    `${SERVER}/train-results/${encodeURIComponent(trainingRunId)}`
  );
  if (!res.ok) return null;
  return await res.json();
}

export async function fetchEvalDetail(evalId) {
  const res = await fetch(
    `${SERVER}/evals/${encodeURIComponent(evalId)}`
  );
  if (!res.ok) return null;
  return await res.json();
}

export async function fetchRunRollouts(trainingRunId, { signal } = {}) {
  const res = await fetch(
    `${SERVER}/runs/${encodeURIComponent(trainingRunId)}/rollouts`,
    { signal },
  );
  if (!res.ok) return [];
  const data = await res.json();
  if (!Array.isArray(data)) return [];
  return data
    .filter((item) => item && typeof item === "object")
    .map((item) => ({
      rollout_id: Number(item.rollout_id) || 0,
      created_at: Number(item.created_at) || 0,
      total: Number(item.total) || 0,
      mean: typeof item.mean === "number" ? item.mean : Number(item.mean) || 0,
      rollout_time: Number.isFinite(Number(item.rollout_time))
        ? Number(item.rollout_time)
        : null,
      error_summary: item.error_summary || null,
    }))
    .sort((a, b) => a.rollout_id - b.rollout_id);
}

export async function fetchRollout(trainingRunId, rolloutId) {
  const res = await fetch(
    `${SERVER}/runs/${encodeURIComponent(trainingRunId)}/rollouts/${encodeURIComponent(rolloutId)}`,
  );
  if (!res.ok) return null;
  return await res.json();
}

// Per-step advantage distribution summaries (one row per training step, each
// carrying that step's overall stats + quantiles). Drives the fan chart that
// shows how the advantage distribution shifts/spreads over training.
export async function fetchRunAdvantages(trainingRunId, { signal } = {}) {
  const res = await fetch(
    `${SERVER}/runs/${encodeURIComponent(trainingRunId)}/advantages`,
    { signal },
  );
  if (!res.ok) return [];
  const data = await res.json();
  if (!Array.isArray(data)) return [];
  return data
    .filter((item) => item && typeof item === "object" && item.stats)
    .map((item) => ({
      rollout_id: Number(item.rollout_id) || 0,
      created_at: Number(item.created_at) || 0,
      num_samples: Number(item.num_samples) || 0,
      num_groups: Number(item.num_groups) || 0,
      stats: item.stats,
    }))
    .sort((a, b) => a.rollout_id - b.rollout_id);
}

// One step's full per-group advantage distribution (for drill-in).
export async function fetchRunAdvantageStep(trainingRunId, rolloutId) {
  const res = await fetch(
    `${SERVER}/runs/${encodeURIComponent(trainingRunId)}/advantages/${encodeURIComponent(rolloutId)}`,
  );
  if (!res.ok) return null;
  return await res.json();
}
