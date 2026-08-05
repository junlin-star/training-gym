<script>
  import { Download, ZoomIn, ZoomOut } from "lucide-svelte";
  import { colorFor, fmtSecs, labelFor, rolloutTimeline } from "../lib/timing.js";

  let {
    stepTimes = null,
    substepTimes = null,
    lanes = null,
    layout = "rows",
    downloadName = "step_substep_times.json",
  } = $props();

  const SUBSTEP_LABELS = {
    evaluate_rollouts: "Eval (before)",
    generate_rollouts: "Generate rollouts",
    offload_rollout: "Offload rollout",
    compute_log_probs: "Compute log probs",
    optimizer_step: "Optimizer step",
    checkpoint_save: "Checkpoint save",
    offload_train: "Offload train",
    weight_sync: "Weight sync",
    evaluate_rollouts_end: "Eval (after)",
  };

  const ORDER = Object.keys(SUBSTEP_LABELS);

  // Timeline zoom bounds: 1 = fit-to-width, MAX_ZOOM = deepest magnification.
  const MIN_ZOOM = 1;
  const MAX_ZOOM = 64;
  const ZOOM_BTN_FACTOR = 1.5;
  const WHEEL_SENSITIVITY = 0.0015;

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

  // One row per phase, each carrying the bars of the times it ran, on the axis
  // this rollout's lanes share; the role a phase was measured on labels the row
  // rather than splitting the rollout into per-role timelines.
  let phaseRows = $derived.by(() => {
    const { rows, span } = rolloutTimeline(lanes);
    const byPhase = new Map();
    for (const row of rows) {
      for (const bar of row) {
        const phase = byPhase.get(bar.name);
        const longest = bar.banded ? bar.longest : bar.duration;
        if (phase) {
          phase.bars.push(bar);
          phase.total += bar.duration;
          phase.longest = Math.max(phase.longest, longest);
          phase.count += bar.count;
        } else {
          byPhase.set(bar.name, {
            name: bar.name,
            role: bar.role,
            bars: [bar],
            total: bar.duration,
            longest,
            count: bar.count,
            start: bar.start,
          });
        }
      }
    }
    return {
      span,
      phases: [...byPhase.values()].sort((a, b) => a.start - b.start),
    };
  });

  // `lanes` present but empty is a real answer -- this rollout recorded no
  // timing -- so it is shown, not passed over to the legacy timeline below.
  let hasMeasuredTiming = $derived(lanes != null);

  function barTitle(phase, bar) {
    const when = `${fmtSecs(bar.start)} \u2192 ${fmtSecs(bar.end)}`;
    if (!bar.banded) {
      return `${labelFor(phase.name)} (${phase.role}): ${fmtSecs(bar.duration)}, ${when}`;
    }
    return `${labelFor(phase.name)} (${phase.role}): ${bar.count} runs over ${when}, ${fmtSecs(bar.duration)} in total, average ${fmtSecs(bar.average)}, longest ${fmtSecs(bar.longest)}`;
  }

  let legend = $derived.by(() => {
    const seen = new Set();
    for (const s of steps) for (const sub of s.substeps) seen.add(sub.name);
    return ORDER.filter((n) => seen.has(n));
  });

  let tip = $state(null);
  let pinned = $state(false);

  // ── Timeline zoom / pan state ────────────────────────────────────────
  let zoom = $state(1);
  let viewport = $state(null);

  function stepWeight(step) {
    const subTotal = step.substeps.reduce((acc, s) => acc + (s.duration ?? 0), 0);
    if (subTotal > 0) return subTotal;
    if (step.duration != null && step.duration > 0) return step.duration;
    return 1;
  }

  function setZoom(next, anchorX = null) {
    const clamped = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, next));
    if (clamped === zoom) return;
    if (viewport) {
      const rect = viewport.getBoundingClientRect();
      const cursorX = anchorX == null ? rect.width / 2 : anchorX - rect.left;
      const contentX = viewport.scrollLeft + cursorX;
      const scale = clamped / zoom;
      zoom = clamped;
      requestAnimationFrame(() => {
        viewport.scrollLeft = contentX * scale - cursorX;
      });
    } else {
      zoom = clamped;
    }
  }

  function handleWheel(e) {
    // Let horizontal trackpad gestures pan natively; vertical wheel zooms.
    if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
    e.preventDefault();
    setZoom(zoom * Math.exp(-e.deltaY * WHEEL_SENSITIVITY), e.clientX);
  }

  // Wheel listeners are passive by default; zooming needs preventDefault.
  function wheelZoom(node) {
    node.addEventListener("wheel", handleWheel, { passive: false });
    return {
      destroy() {
        node.removeEventListener("wheel", handleWheel);
      },
    };
  }

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

{#snippet segment(step, sub)}
  <div
    class="seg"
    class:seg-null={sub.duration == null}
    class:active={pinned && isActive(step, sub)}
    style:flex-grow={sub.duration == null ? undefined : Math.max(sub.duration, 0.01)}
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

{#if hasMeasuredTiming}
  <div class="step-timings">
    {#if phaseRows.phases.length === 0}
      <div class="no-timing">no timing recorded for this rollout</div>
    {:else}
      <div class="lane-phases">
        {#each phaseRows.phases as phase (phase.name)}
          <div class="phase-row">
            <span class="phase-name" style:color={colorFor(phase.name)}>
              {labelFor(phase.name)}
            </span>
            <!-- The driver is where a phase lives unless it says otherwise. -->
            <span class="phase-role">{phase.role === "driver" ? "" : phase.role}</span>
            <span class="phase-track">
              {#each phase.bars as bar (bar.key)}
                <span
                  class="phase-bar"
                  class:banded={bar.banded}
                  style:background={colorFor(phase.name)}
                  style:left={`${(bar.start / phaseRows.span) * 100}%`}
                  style:width={`${Math.max(((bar.end - bar.start) / phaseRows.span) * 100, 0.6)}%`}
                  title={barTitle(phase, bar)}
                ></span>
              {/each}
            </span>
            <span class="phase-total">{fmtSecs(phase.total)}</span>
            <span class="phase-count">
              {#if phase.count > 1}
                {phase.count}× · longest {fmtSecs(phase.longest)} · average
                {fmtSecs(phase.total / phase.count)}
              {/if}
            </span>
          </div>
        {/each}
      </div>
    {/if}
  </div>
{:else if hasData}
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
          <div class="tl-toolbar">
            <div class="zoom-controls">
              <button
                class="zoom-btn"
                onclick={() => setZoom(zoom / ZOOM_BTN_FACTOR)}
                disabled={zoom <= MIN_ZOOM}
                title="Zoom out"
              >
                <ZoomOut size={13} />
              </button>
              <button
                class="zoom-level"
                onclick={() => setZoom(MIN_ZOOM)}
                disabled={zoom <= MIN_ZOOM}
                title="Reset zoom to fit"
              >
                {zoom >= 10 ? Math.round(zoom) : zoom.toFixed(1).replace(/\.0$/, "")}×
              </button>
              <button
                class="zoom-btn"
                onclick={() => setZoom(zoom * ZOOM_BTN_FACTOR)}
                disabled={zoom >= MAX_ZOOM}
                title="Zoom in"
              >
                <ZoomIn size={13} />
              </button>
            </div>
            <button
              class="dl-btn"
              onclick={downloadJson}
              title="Download step + substep times as JSON"
            >
              <Download size={13} />
              Download JSON
            </button>
          </div>
        {/if}
      </div>
    {/if}

    {#if layout === "timeline"}
      <div class="tl-viewport" bind:this={viewport} use:wheelZoom>
        <div class="tl-track" style:width={`${zoom * 100}%`}>
          {#each steps as step (step.key)}
            <div class="tl-step" style:flex-grow={stepWeight(step)}>
              <div class="tl-step-head">
                <span class="tl-step-name">Step {step.n}</span>
                <span class="tl-step-dur">{fmtSecs(step.duration)}</span>
              </div>
              {#if step.substeps.length}
                <div class="bar tl-bar">
                  {#each step.substeps as sub (sub.name)}
                    {@render segment(step, sub)}
                  {/each}
                </div>
              {:else}
                <div class="bar tl-bar bar-empty"></div>
              {/if}
            </div>
          {/each}
        </div>
      </div>
      <div class="tl-hint">Scroll to zoom · shift-scroll or drag the scrollbar to pan</div>
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
                {@render segment(step, sub)}
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

  .no-timing {
    color: var(--color-c-gray-45, #6e6e6e);
    font-size: 0.85rem;
    padding: 4px 0;
  }

  .lane-phases {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .phase-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.85rem;
  }

  .phase-name {
    min-width: 10rem;
    flex: 0 0 auto;
  }

  .phase-track {
    position: relative;
    flex: 1 1 auto;
    min-width: 4rem;
    height: 10px;
    border-radius: 3px;
    background: var(--color-c-gray-08, #1c1c1c);
  }

  .phase-bar {
    position: absolute;
    top: 0;
    height: 100%;
    border-radius: 3px;
    min-width: 2px;
  }

  /* A phase that ran too many times to keep each run: one band over the span
     it covered, hatched so it doesn't read as a single continuous run. */
  .phase-bar.banded {
    background-image: repeating-linear-gradient(
      45deg,
      rgba(0, 0, 0, 0.28),
      rgba(0, 0, 0, 0.28) 3px,
      transparent 3px,
      transparent 6px
    );
  }

  .phase-role {
    color: var(--color-c-gray-45, #6e6e6e);
    flex: 0 0 4rem;
    font-size: 0.78rem;
  }

  .phase-total {
    font-variant-numeric: tabular-nums;
    flex: 0 0 auto;
  }

  .phase-count {
    color: var(--color-c-gray-45, #6e6e6e);
    font-variant-numeric: tabular-nums;
    flex: 0 0 9rem;
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

  /* ── Timeline (full-width zoomable bar across all steps) ─────────────── */
  .tl-toolbar {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
  }

  .zoom-controls {
    display: inline-flex;
    align-items: center;
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 4px;
    overflow: hidden;
  }

  .zoom-btn,
  .zoom-level {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: none;
    border: none;
    color: var(--muted);
    font-size: 11px;
    padding: 3px 7px;
    cursor: pointer;
  }

  .zoom-level {
    min-width: 38px;
    border-left: 1px solid var(--border, #2f2f2f);
    border-right: 1px solid var(--border, #2f2f2f);
    font-variant-numeric: tabular-nums;
  }

  .zoom-btn:hover:not(:disabled),
  .zoom-level:hover:not(:disabled) {
    color: var(--text);
    background: var(--color-c-gray-08, #1c1c1c);
  }

  .zoom-btn:disabled,
  .zoom-level:disabled {
    opacity: 0.4;
    cursor: default;
  }

  .tl-viewport {
    overflow-x: auto;
    overflow-y: hidden;
    padding-bottom: 6px;
    overscroll-behavior-x: contain;
    touch-action: pan-x;
    -webkit-overflow-scrolling: touch;
  }

  .tl-track {
    display: flex;
    gap: 3px;
    min-width: 100%;
  }

  .tl-step {
    display: flex;
    flex-direction: column;
    gap: 3px;
    flex-basis: 0;
    overflow: hidden;
  }

  @media (max-width: 900px) {
    .tl-step {
      min-width: 72px;
    }
  }

  .tl-step-head {
    display: flex;
    align-items: baseline;
    gap: 6px;
    font-size: 10px;
    line-height: 14px;
    white-space: nowrap;
    overflow: hidden;
  }

  .tl-step-name {
    color: var(--text-bright);
    font-weight: 500;
  }

  .tl-step-dur {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .tl-bar {
    height: 18px;
  }

  .tl-hint {
    font-size: 10px;
    color: var(--muted);
    opacity: 0.7;
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
