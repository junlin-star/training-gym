<script>
  import StatusPill from "./StatusPill.svelte";
  import FrameworkStageProgress from "./FrameworkStageProgress.svelte";
  import TimeAgo from "./TimeAgo.svelte";
  import { smoothedStageLabel } from "../lib/format.js";

  // The run-summary block shared by the list drawer and the detail page's
  // Summary tab, so both render identical metadata: status, stage, model,
  // dataset, recipe, timing, the full Slime parameter dump, and the tuned
  // recipe fields.
  let { run, getStatus, showFrameworkStatus, getFrameworkStatus, modelName, fmtDuration } =
    $props();

  let recipe = $derived.by(() => run?.config?.recipe || run?.config?.preset || {});
  let recipeEntries = $derived.by(() =>
    Object.entries(recipe).filter(
      ([, value]) => value !== undefined && value !== null && String(value) !== "",
    ),
  );
  let recipeJson = $derived.by(() =>
    Object.keys(recipe).length ? JSON.stringify(recipe, null, 2) : "",
  );

  function isSlimeRun() {
    return String(run?.framework || "").toLowerCase() === "slime";
  }

  function frameworkProgress() {
    const p = run?.framework_progress;
    if (!p || typeof p !== "object") return null;
    const current = Number(p.current);
    const total = Number(p.total);
    if (!Number.isFinite(current) || !Number.isFinite(total) || total <= 0) return null;
    return {
      current: Math.max(0, Math.min(current, total)),
      total,
      unit: p.unit || "step",
    };
  }

  function frameworkStatusLabel() {
    return smoothedStageLabel(getFrameworkStatus?.(run), run?.framework_progress);
  }

  function progressLabel(progress) {
    if (!progress) return "";
    const unit = String(progress.unit || "step");
    const label = unit.charAt(0).toUpperCase() + unit.slice(1);
    return `${label} ${progress.current} / ${progress.total}`;
  }

  function runDuration() {
    if (!run) return "—";
    if (typeof run.duration_seconds === "number" && run.duration_seconds >= 0) {
      return fmtDuration(0, run.duration_seconds);
    }
    if (run.started_at) return fmtDuration(run.started_at, run.ended_at);
    return "—";
  }

  function formatFieldLabel(field) {
    return field.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
  }

  function formatFieldValue(value) {
    if (typeof value === "number") {
      return Number.isInteger(value) ? String(value) : value.toExponential(1);
    }
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  }
</script>

{#if run}
  <div class="run-summary">
    <section class="summary-section">
      <div class="kv">
        <span class="kv-key">Status</span>
        <StatusPill status={getStatus(run)} />
      </div>
      {#if showFrameworkStatus(run) && frameworkStatusLabel()}
        {@const progress = frameworkProgress()}
        <div class="kv">
          <span class="kv-key">Stage</span>
          <FrameworkStageProgress
            {progress}
            progressLabel={progressLabel(progress)}
            stageLabel={frameworkStatusLabel()}
            active={getStatus(run).toLowerCase() === "running"}
          />
        </div>
      {/if}
      <div class="kv">
        <span class="kv-key">Model</span>
        <span class="kv-value">{modelName(run)}</span>
      </div>
      <div class="kv">
        <span class="kv-key">Dataset</span>
        <span class="kv-value">{run.config_summary?.dataset_name || "—"}</span>
      </div>
      <div class="kv">
        <span class="kv-key">Recipe</span>
        <span class="kv-value">{run.framework || "—"}</span>
      </div>
      <div class="kv">
        <span class="kv-key">Duration</span>
        <span class="kv-value">{runDuration()}</span>
      </div>
      <div class="kv">
        <span class="kv-key">Started</span>
        <span class="kv-value">
          <TimeAgo timestamp={run.started_at || run.created_at} showJustNow />
        </span>
      </div>
      <div class="kv">
        <span class="kv-key">Last updated</span>
        <span class="kv-value">
          <TimeAgo timestamp={run.updated_at} showJustNow falsyRepresentation="—" />
        </span>
      </div>
    </section>

    {#if isSlimeRun() && recipeJson}
      <section class="summary-section">
        <h3 class="summary-section-title">Full Slime parameters</h3>
        <pre class="summary-json">{recipeJson}</pre>
      </section>
    {/if}

    <section class="summary-section">
      <h3 class="summary-section-title">Training recipe</h3>
      {#if recipeEntries.length}
        {#each recipeEntries as [field, value] (field)}
          <div class="kv">
            <span class="kv-key">{formatFieldLabel(field)}</span>
            <span class="kv-value kv-value-mono">{formatFieldValue(value)}</span>
          </div>
        {/each}
      {:else}
        <div class="summary-empty">No recipe values found for this run.</div>
      {/if}
    </section>
  </div>
{/if}

<style>
  .summary-section {
    border-bottom: 1px solid var(--color-c-gray-10, #2f2f2f);
    padding: 16px 0;
  }

  .summary-section:first-child {
    padding-top: 0;
  }

  .summary-section:last-child {
    border-bottom: 0;
  }

  .summary-section-title {
    color: var(--text-bright);
    font-size: 14px;
    font-weight: 500;
    line-height: 20px;
    margin-bottom: 8px;
  }

  .kv {
    display: grid;
    grid-template-columns: 100px minmax(0, 1fr);
    align-items: baseline;
    gap: 8px;
    padding: 4px 0;
  }

  .kv-key {
    color: var(--muted);
    font-size: 12px;
    line-height: 16px;
  }

  .kv-value {
    color: var(--text);
    font-size: 14px;
    line-height: 20px;
    overflow-wrap: anywhere;
  }

  .kv-value-mono {
    font-family: var(--font-mono);
    font-size: 12px;
    line-height: 16px;
  }

  .summary-empty {
    color: var(--muted);
    font-size: 12px;
    line-height: 16px;
  }

  .summary-json {
    border: 1px solid var(--color-c-gray-10, #2f2f2f);
    border-radius: 8px;
    background: color-mix(in srgb, var(--panel-alt) 74%, black);
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 16px;
    margin: 0;
    max-height: 360px;
    overflow: auto;
    padding: 10px;
    white-space: pre;
  }
</style>
