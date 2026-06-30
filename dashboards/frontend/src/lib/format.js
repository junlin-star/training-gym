function unwrapValue(value) {
  if (value && typeof value === "object" && "value" in value) return value.value;
  return value;
}

export function toEpochSeconds(value) {
  const ts = unwrapValue(value);
  if (ts == null || ts === "") return null;

  if (typeof ts === "number") {
    if (!Number.isFinite(ts)) return null;
    if (ts > 1e12) return ts / 1000;
    return ts;
  }

  if (typeof ts === "string") {
    const trimmed = ts.trim();
    if (!trimmed) return null;

    const numeric = Number(trimmed);
    if (Number.isFinite(numeric)) return toEpochSeconds(numeric);

    const parsedMs = Date.parse(trimmed);
    if (!Number.isNaN(parsedMs)) return parsedMs / 1000;
    return null;
  }

  if (ts instanceof Date) {
    const ms = ts.getTime();
    if (Number.isNaN(ms)) return null;
    return ms / 1000;
  }

  return null;
}

export function fmtDate(ts) {
  const seconds = toEpochSeconds(ts);
  if (seconds == null) return "—";
  const d = new Date(seconds * 1000);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toISOString().replace("T", " ").slice(0, 19) + " UTC";
}

export function fmtDuration(start, end) {
  const startTs = toEpochSeconds(start);
  if (startTs == null) return "—";
  const endTs = toEpochSeconds(end) ?? Date.now() / 1000;
  let secs = Math.max(0, Math.floor(endTs - startTs));
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function fmtCluster(summary) {
  if (!summary) return "—";
  const gpu = summary.gpu_type || "?";
  const nodes = summary.actor_num_nodes || 0;
  const gpusPerNode = summary.actor_num_gpus_per_node || 0;
  if (!nodes && !gpusPerNode) return gpu || "—";
  const totalGpus = nodes * gpusPerNode;
  return `${totalGpus}x ${gpu}`;
}

export function fmtLr(lr) {
  if (!lr) return "—";
  if (lr < 0.001) return lr.toExponential(1);
  return String(lr);
}

export function truncateId(id) {
  if (!id) return "—";
  if (id.length <= 12) return id;
  return id.slice(0, 12) + "…";
}

export function getGroupTags(run) {
  const tags = run?.metadata?.group_tags;
  const groupId = tags?.group_id || run?.group_id || run?.metadata?.group_id || "";
  if (!groupId && (!tags || typeof tags !== "object")) return null;

  const overrides =
    tags?.overrides && typeof tags.overrides === "object" ? tags.overrides : {};
  const rawTags = Array.isArray(tags?.tags) ? tags.tags : [];
  const displayTags = rawTags.length
    ? rawTags
    : Object.entries(overrides).map(([key, value]) => ({
        key,
        label: key.split(".").at(-1)?.replace(/_/g, " ") || key,
        value,
      }));

  return {
    group_id: groupId,
    axes: Array.isArray(tags?.axes) ? tags.axes : Object.keys(overrides),
    overrides,
    tags: displayTags,
  };
}

export function formatTagValue(value) {
  if (value == null) return "—";
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

const STAGE_LABELS = {
  initializing: "Initializing",
  download_model: "Downloading model",
  convert_model: "Converting model",
  prepare_dataset: "Preparing dataset",
  initialize_rollouts: "Initializing rollouts",
  generate_rollouts: "Generating rollouts",
  evaluate_rollouts: "Evaluating rollouts",
  compute_log_probs: "Computing log probs",
  optimizer_step: "Optimizer step",
  weight_sync: "Weight sync",
  offload_rollout: "Offload rollout",
  offload_train: "Offload train",
  checkpoint_save: "Saving checkpoint",
  training: "Training",
};

export function formatStageLabel(status) {
  const raw = String(status || "").trim();
  if (!raw) return "";
  const normalized = raw.toLowerCase();
  return (
    STAGE_LABELS[normalized] ||
    normalized.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
  );
}

// Stages where common/train.py marks the stage *before* the Modal function
// runs; the dashboard distinguishes "queuing for a GPU" from "actually
// running" via metadata.framework_progress.is_active.
const QUEUEABLE_STAGES = new Set(["download_model", "convert_model"]);

export function isQueuingStage(status, progress) {
  if (!QUEUEABLE_STAGES.has(String(status || "").toLowerCase())) return false;
  if (!progress || typeof progress !== "object") return false;
  // Explicit false from server; treat undefined as "running" so legacy data
  // doesn't suddenly start showing "Queuing".
  return progress.is_active === false;
}

// Top-level stage label. We surface the *actual* framework stage — including
// inner-loop phases like "Weight sync" / "Optimizer step" — rather than
// collapsing them behind a generic "Training" label, so the user can see
// exactly what the run is doing. The per-step progress bar/counter is shown
// alongside this (via framework_progress), so the granular stage and the
// overall step progress are both visible. Still surfaces "Queuing for GPU"
// when a download/convert stage is marked but the Modal function hasn't
// started executing yet.
export function smoothedStageLabel(status, progress) {
  if (isQueuingStage(status, progress)) {
    return `Queuing for GPU — ${formatStageLabel(status)}`;
  }
  return formatStageLabel(status);
}
