<script>
  import { Download } from "lucide-svelte";

  let {
    stepTimes = null,
    substepTimes = null,
    layout = "rows",
    downloadName = "step_substep_times.json",
  } = $props();

  const SUBSTEP_LABELS = {
    evaluate_rollouts: "Eval (before)",
    generate_rollouts: "Generate rollouts",
    offload_rollout: "Offload rollout",
    compute_log_probs: "Compute log probs",
    optimizer_step: "Optimizer step",
    weight_sync: "Weight sync",
    checkpoint_save: "Checkpoint save",
    offload_train: "Offload train",
    evaluate_rollouts_end: "Eval (after)",
  };

  const SUBSTEP_COLORS = {
    evaluate_rollouts: "#60a5fa",
    generate_rollouts: "#34d399",
    offload_rollout: "#a78bfa",
    compute_log_probs: "#fbbf24",
    optimizer_step: "#f87171",
    weight_sync: "#22d3ee",
    checkpoint_save: "#f472b6",
    offload_train: "#c084fc",
    evaluate_rollouts_end: "#818cf8",
  };

  const ORDER = Object.keys(SUBSTEP_LABELS);

  // Timeline horizontal scale (pixels per second) + fixed width for dropped substeps.
  const PX_PER_SEC = 2;
  const NULL_WIDTH = 8;

  function labelFor(name) {
    return SUBSTEP_LABELS[name] || name.replace(/_/g, " ");
  }

  function colorFor(name) {
    return SUBSTEP_COLORS[name] || "var(--color-c-gray-40, #5e5e5e)";
  }

  // Durations are float seconds; keep up to 3 decimals (trailing zeros trimmed).
  function fmtSecs(s) {
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

  function downloadJson() {
    const payload = { step_times: stepTimes || {}, substep_times: substepTimes || {} };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = downloadName;
    a.click();
    URL.revokeObjectURL(url);
  }

  function segWidth(sub) {
    if (sub.duration == null) return NULL_WIDTH;
    return Math.max(sub.duration * PX_PER_SEC, 2);
  }

  let steps = $derived.by(() => {
    const stepKeys = Object.keys(stepTimes || {});
    const subKeys = Object.keys(substepTimes || {});
    const keys = Array.from(new Set([...stepKeys, ...subKeys]));
    const out = keys.map((k) => {
      const st = (stepTimes || {})[k] || null;
      const subs = (substepTimes || {})[k] || {};
      const substeps = Object.entries(subs)
        .map(([name, v]) => ({
          name,
          start: v?.start ?? null,
          duration: v?.duration_s ?? null,
        }))
        .sort((a, b) => (a.start ?? 0) - (b.start ?? 0));
      return {
        key: k,
        n: Number.isFinite(Number(k)) ? Number(k) : k,
        duration: st?.duration_s ?? null,
        substeps,
      };
    });
    out.sort((a, b) => (Number(a.key) || 0) - (Number(b.key) || 0));
    return out;
  });

  let hasData = $derived(steps.length > 0);

  let legend = $derived.by(() => {
    const seen = new Set();
    for (const s of steps) for (const sub of s.substeps) seen.add(sub.name);
    return ORDER.filter((n) => seen.has(n));
  });

  let tip = $state(null);
  let pinned = $state(false);

  function isActive(step, sub) {
    return tip && tip.step === step.n && tip.name === sub.name;
  }

  function showTip(e, step, sub) {
    if (pinned) return;
    tip = { x: e.clientX, y: e.clientY, step: step.n, name: sub.name, dur: sub.duration };
  }

  function moveTip(e) {
    if (pinned || !tip) return;
    tip = { ...tip, x: e.clientX, y: e.clientY };
  }

  function hideTip() {
    if (pinned) return;
    tip = null;
  }

  function pinTip(e, step, sub) {
    e.stopPropagation();
    if (pinned && isActive(step, sub)) {
      pinned = false;
      tip = null;
      return;
    }
    pinned = true;
    tip = { x: e.clientX, y: e.clientY, step: step.n, name: sub.name, dur: sub.duration };
  }

  function clearPin() {
    if (!pinned) return;
    pinned = false;
    tip = null;
  }
</script>

<svelte:window onclick={clearPin} />

{#snippet segment(step, sub, timeline)}
  <div
    class="seg"
    class:seg-null={sub.duration == null}
    class:active={pinned && isActive(step, sub)}
    style:flex-grow={timeline || sub.duration == null ? undefined : Math.max(sub.duration, 0.01)}
    style:width={timeline ? `${segWidth(sub)}px` : undefined}
    style:background={sub.duration == null ? undefined : colorFor(sub.name)}
    role="button"
    tabindex="0"
    onmouseenter={(e) => showTip(e, step, sub)}
    onmousemove={moveTip}
    onmouseleave={hideTip}
    onclick={(e) => pinTip(e, step, sub)}
    onkeydown={(e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        pinTip(e, step, sub);
      }
    }}
  ></div>
{/snippet}

{#if hasData}
  <div class="step-timings">
    {#if legend.length || layout === "timeline"}
      <div class="legend-row">
        <div class="legend">
          {#each legend as name (name)}
            <span class="legend-item">
              <span class="swatch" style:background={colorFor(name)}></span>
              {labelFor(name)}
            </span>
          {/each}
        </div>
        {#if layout === "timeline"}
          <button
            class="dl-btn"
            onclick={downloadJson}
            title="Download step + substep times as JSON"
          >
            <Download size={13} />
            Download JSON
          </button>
        {/if}
      </div>
    {/if}

    {#if layout === "timeline"}
      <div class="timeline-scroll">
        <div class="bar tl-bar">
          {#each steps as step, si (step.key)}
            {#if si > 0}
              <div class="tl-divider" title={`Step ${step.n}`}></div>
            {/if}
            {#each step.substeps as sub (sub.name)}
              {@render segment(step, sub, true)}
            {/each}
          {/each}
        </div>
      </div>
    {:else}
      {#each steps as step (step.key)}
        <div class="step-row">
          <div class="step-head">
            <span class="step-name">Step {step.n}</span>
            <span class="step-dur">{fmtSecs(step.duration)}</span>
          </div>
          {#if step.substeps.length}
            <div class="bar">
              {#each step.substeps as sub (sub.name)}
                {@render segment(step, sub, false)}
              {/each}
            </div>
          {:else}
            <div class="bar bar-empty"></div>
          {/if}
        </div>
      {/each}
    {/if}
  </div>

  {#if tip}
    <div class="tg-tip" class:pinned style:left={`${tip.x}px`} style:top={`${tip.y}px`}>
      <span class="tg-tip-step">Step {tip.step}</span>
      <span class="tg-tip-name">{labelFor(tip.name)}</span>
      <span class="tg-tip-dur">
        {tip.dur == null ? "unknown (report dropped)" : fmtSecs(tip.dur)}
      </span>
    </div>
  {/if}
{/if}

<style>
  .step-timings {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .legend-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 4px;
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 14px;
    font-size: 11px;
    color: var(--muted);
  }

  .dl-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    flex-shrink: 0;
    background: none;
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 4px;
    color: var(--muted);
    font-size: 11px;
    padding: 3px 8px;
    cursor: pointer;
  }

  .dl-btn:hover {
    color: var(--text);
    border-color: var(--border-strong, #4a4a4a);
  }

  .legend-item {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }

  .swatch {
    width: 9px;
    height: 9px;
    border-radius: 2px;
    flex-shrink: 0;
  }

  .step-row {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .step-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
    font-size: 12px;
  }

  .step-name {
    color: var(--text-bright);
    font-weight: 500;
  }

  .step-dur {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .bar {
    display: flex;
    height: 14px;
    border-radius: 3px;
    overflow: hidden;
    background: var(--color-c-gray-08, #1c1c1c);
    gap: 1px;
  }

  .bar-empty {
    background: var(--color-c-gray-10, #2f2f2f);
  }

  .seg {
    min-width: 2px;
    height: 100%;
    cursor: pointer;
    transition: filter 0.1s ease;
  }

  .seg:hover {
    filter: brightness(1.25);
  }

  .seg.active {
    outline: 1px solid var(--text-bright, #fff);
    outline-offset: -1px;
    filter: brightness(1.3);
  }

  /* Dropped substep: visible but doesn't distort the proportional widths. */
  .seg-null {
    flex: 0 0 16px;
    background: repeating-linear-gradient(
      45deg,
      var(--color-c-gray-20, #3a3a3a),
      var(--color-c-gray-20, #3a3a3a) 3px,
      var(--color-c-gray-10, #2f2f2f) 3px,
      var(--color-c-gray-10, #2f2f2f) 6px
    );
  }

  /* ── Timeline (one long continuous bar across all steps) ─────────────── */
  .timeline-scroll {
    overflow-x: auto;
    overflow-y: hidden;
    padding-bottom: 6px;
  }

  .tl-bar {
    width: max-content;
    min-width: 100%;
    height: 18px;
  }

  .tl-bar .seg {
    flex: 0 0 auto;
  }

  /* Thin marker between steps within the single continuous bar. */
  .tl-divider {
    flex: 0 0 2px;
    height: 100%;
    background: var(--color-c-gray-02, #0d0d0d);
  }

  /* ── Pinnable tooltip ────────────────────────────────────────────────── */
  .tg-tip {
    position: fixed;
    z-index: 1000;
    transform: translate(-50%, calc(-100% - 10px));
    pointer-events: none;
    display: flex;
    flex-direction: column;
    gap: 1px;
    padding: 6px 9px;
    border-radius: 6px;
    background: var(--color-c-gray-02, #0d0d0d);
    border: 1px solid var(--border, #3a3a3a);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
    font-size: 11px;
    white-space: nowrap;
  }

  .tg-tip.pinned {
    border-color: var(--accent, #60a5fa);
  }

  .tg-tip-step {
    color: var(--muted);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .tg-tip-name {
    color: var(--text-bright, #fff);
    font-weight: 600;
  }

  .tg-tip-dur {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }
</style>
