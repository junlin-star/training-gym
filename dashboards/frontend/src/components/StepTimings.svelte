<script>
  import { Download, ZoomIn, ZoomOut } from "lucide-svelte";

  let {
    stepTimes = null,
    substepTimes = null,
    substepTimingIntervals = null,
    layout = "rows",
    asyncMode = false,
    downloadName = "step_substep_times.json",
  } = $props();

  let detailedTimingIntervals = $derived(substepTimingIntervals);

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

  const ASYNC_SUBSTEP_LABELS = {
    ...SUBSTEP_LABELS,
    generate_rollouts: "Rollout + reward",
    training: "Training",
    train_model: "Forward / backward",
  };

  const ASYNC_SUBSTEP_COLORS = {
    ...SUBSTEP_COLORS,
    training: "var(--color-c-dataviz-primary-7, #648fe0)",
    train_model: "var(--color-c-dataviz-paired-4, #6cabc1)",
    optimizer_step: "var(--color-c-dataviz-paired-7, #8956fa)",
  };

  const SUBSTEP_ORDER = Object.keys(SUBSTEP_LABELS);
  const ASYNC_SUBSTEP_ORDER = [
    "evaluate_rollouts",
    "generate_rollouts",
    "training",
    "train_model",
    "optimizer_step",
    "checkpoint_save",
    "weight_sync",
    "offload_rollout",
    "compute_log_probs",
    "offload_train",
    "evaluate_rollouts_end",
  ];

  // Timeline zoom bounds: 1 = fit-to-width, MAX_ZOOM = deepest magnification.
  const MIN_ZOOM = 1;
  const MAX_ZOOM = 64;
  const ZOOM_BTN_FACTOR = 1.5;
  const WHEEL_SENSITIVITY = 0.0015;

  function labelFor(name) {
    return SUBSTEP_LABELS[name] || name.replace(/_/g, " ");
  }

  function colorFor(name) {
    return SUBSTEP_COLORS[name] || "var(--color-c-gray-40, #5e5e5e)";
  }

  function asyncLabelFor(name) {
    return ASYNC_SUBSTEP_LABELS[name] || labelFor(name);
  }

  function asyncColorFor(name) {
    return ASYNC_SUBSTEP_COLORS[name] || colorFor(name);
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
    const payload = {
      step_times: stepTimes || {},
      substep_times: substepTimes || {},
    };
    if (detailedTimingIntervals) {
      payload.substep_timing_intervals = detailedTimingIntervals;
      payload.substep_spans = detailedTimingIntervals;
    }
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
        .map(([name, v]) => {
          const detailed = detailedTimingIntervals?.[k]?.[name];
          const hasDetails = Array.isArray(detailed) && detailed.length > 0;
          const values = hasDetails ? detailed : [v];
          const segments = [];
          for (const [index, value] of values.entries()) {
            if (value?.start == null || value?.duration_s == null) continue;
            const start = Number(value?.start);
            const duration = Number(value?.duration_s);
            if (!Number.isFinite(start) || !Number.isFinite(duration) || duration < 0) {
              continue;
            }
            segments.push({
              innerStep: hasDetails ? (value.step_id ?? index) : null,
              start,
              duration,
              end: start + duration,
            });
          }
          return {
            name,
            start: v?.start ?? null,
            duration: v?.duration_s ?? null,
            segments,
          };
        })
        .sort((a, b) => (a.start ?? 0) - (b.start ?? 0));
      const step = {
        key: k,
        n: Number.isFinite(Number(k)) ? Number(k) : k,
        rolloutId: Number.isFinite(Number(k)) ? Math.max(0, Number(k) - 1) : k,
        duration: st?.duration_s ?? null,
        substeps,
      };
      step.timeline = [];
      for (const sub of substeps) {
        for (const item of sub.segments) {
          step.timeline.push({
            step,
            sub: { name: sub.name, ...item },
            start: item.start,
            end: item.end,
          });
        }
      }
      return step;
    });
    out.sort((a, b) => (Number(a.key) || 0) - (Number(b.key) || 0));
    return out;
  });

  let hasData = $derived(steps.length > 0);

  const TRAINING_CHILDREN = new Set(["train_model", "optimizer_step"]);

  function tooltipLabel(step, sub) {
    const repeated =
      step.timeline.filter((interval) => interval.sub.name === sub.name).length > 1;
    if (repeated && TRAINING_CHILDREN.has(sub.name) && sub.innerStep != null) {
      return `${asyncLabelFor(sub.name)} ${sub.innerStep + 1}`;
    }
    return asyncMode ? asyncLabelFor(sub.name) : labelFor(sub.name);
  }

  function trainingUpdates(step) {
    return step.timeline
      .filter((interval) => TRAINING_CHILDREN.has(interval.sub.name))
      .sort((a, b) => a.start - b.start);
  }

  function topLevelSubsteps(step) {
    const hasUpdates = trainingUpdates(step).length > 0;
    return hasUpdates
      ? step.substeps.filter(
          (sub) => sub.name !== "training" && !TRAINING_CHILDREN.has(sub.name),
        )
      : step.substeps;
  }

  let asyncTimeline = $derived.by(() => {
    const rollout = [];
    const training = [];
    const trainingWindows = [];
    const coordination = [];
    for (const step of steps) {
      const hasUpdates = trainingUpdates(step).length > 0;
      for (const span of step.timeline) {
        if (TRAINING_CHILDREN.has(span.sub.name)) training.push(span);
        else if (span.sub.name === "training" && hasUpdates) trainingWindows.push(span);
        else if (
          span.sub.name === "generate_rollouts" ||
          span.sub.name.startsWith("evaluate_rollouts")
        ) {
          rollout.push(span);
        } else if (span.sub.name === "training") training.push(span);
        else coordination.push(span);
      }
    }
    const spans = [...rollout, ...trainingWindows, ...training, ...coordination];
    if (!spans.length) {
      return { start: 0, duration: 1, rollout, training, trainingWindows, coordination };
    }
    const start = Math.min(...spans.map((span) => span.start));
    const end = Math.max(...spans.map((span) => span.end));
    return {
      start,
      duration: Math.max(end - start, 0.001),
      rollout,
      training,
      trainingWindows,
      coordination,
    };
  });

  let legend = $derived.by(() => {
    const seen = new Set();
    if (asyncMode) {
      for (const step of steps) {
        for (const sub of topLevelSubsteps(step)) seen.add(sub.name);
        for (const span of trainingUpdates(step)) seen.add(span.sub.name);
      }
      if (asyncTimeline.trainingWindows.length) seen.add("training");
    } else {
      for (const step of steps) for (const sub of step.substeps) seen.add(sub.name);
    }
    const order = asyncMode ? ASYNC_SUBSTEP_ORDER : SUBSTEP_ORDER;
    return order.filter((n) => seen.has(n));
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
    return (
      tip &&
      tip.rolloutId === step.rolloutId &&
      tip.name === sub.name &&
      tip.innerStep === (sub.innerStep ?? null)
    );
  }

  function showTip(e, step, sub) {
    if (pinned) return;
    tip = {
      x: e.clientX,
      y: e.clientY,
      rolloutId: step.rolloutId,
      innerStep: sub.innerStep ?? null,
      name: sub.name,
      label: tooltipLabel(step, sub),
      stepLabel: asyncMode ? `Rollout ${step.rolloutId}` : `Step ${step.n}`,
      dur: sub.duration,
    };
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
    tip = {
      x: e.clientX,
      y: e.clientY,
      rolloutId: step.rolloutId,
      innerStep: sub.innerStep ?? null,
      name: sub.name,
      label: tooltipLabel(step, sub),
      stepLabel: asyncMode ? `Rollout ${step.rolloutId}` : `Step ${step.n}`,
      dur: sub.duration,
    };
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

{#snippet asyncSegment(span)}
  {@const trainingChild = TRAINING_CHILDREN.has(span.sub.name)}
  <div
    class="seg async-seg"
    class:training-inner-seg={trainingChild}
    class:active={pinned && isActive(span.step, span.sub)}
    style={`left:${((span.start - asyncTimeline.start) / asyncTimeline.duration) * 100}%;width:${(span.sub.duration / asyncTimeline.duration) * 100}%;background:${asyncColorFor(span.sub.name)}`}
    role="button"
    tabindex="0"
    onmouseenter={(e) => showTip(e, span.step, span.sub)}
    onmousemove={moveTip}
    onmouseleave={hideTip}
    onclick={(e) => pinTip(e, span.step, span.sub)}
    onkeydown={(e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        pinTip(e, span.step, span.sub);
      }
    }}
  ></div>
{/snippet}

{#snippet trainingWindow(span)}
  <div
    class="training-window"
    class:active={pinned && isActive(span.step, span.sub)}
    style={`left:${((span.start - asyncTimeline.start) / asyncTimeline.duration) * 100}%;width:${(span.sub.duration / asyncTimeline.duration) * 100}%;--training-color:${asyncColorFor(span.sub.name)}`}
    role="button"
    tabindex="0"
    onmouseenter={(e) => showTip(e, span.step, span.sub)}
    onmousemove={moveTip}
    onmouseleave={hideTip}
    onclick={(e) => pinTip(e, span.step, span.sub)}
    onkeydown={(e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        pinTip(e, span.step, span.sub);
      }
    }}
  ></div>
{/snippet}

{#snippet asyncLane(label, spans, windows = [])}
  {#if spans.length || windows.length}
    <div class="async-lane">
      <div class="async-lane-label tl-step-name">{label}</div>
      <div class="bar tl-bar async-lane-track">
        {#each windows as span (`window-${span.step.key}-${span.start}`)}
          {@render trainingWindow(span)}
        {/each}
        {#each spans as span (`${span.step.key}-${span.sub.name}-${span.sub.innerStep ?? "total"}-${span.start}`)}
          {@render asyncSegment(span)}
        {/each}
      </div>
    </div>
  {/if}
{/snippet}

{#if hasData}
  <div class="step-timings">
    {#if legend.length || layout === "timeline"}
      <div class="legend-row">
        <div class="legend">
          {#each legend as name (name)}
            <span class="legend-item">
              <span class="swatch" style:background={asyncMode ? asyncColorFor(name) : colorFor(name)}></span>
              {asyncMode ? asyncLabelFor(name) : labelFor(name)}
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

    {#if layout === "timeline" && asyncMode}
      <div class="tl-viewport" bind:this={viewport} use:wheelZoom>
        <div class="async-timeline" style:width={`${zoom * 100}%`}>
          <div class="async-axis">
            <span class="tl-step-name">Async wall-clock timeline</span>
            <span class="tl-step-dur">{fmtSecs(asyncTimeline.duration)}</span>
          </div>
          {@render asyncLane("Rollout + reward", asyncTimeline.rollout)}
          {@render asyncLane("Training", asyncTimeline.training, asyncTimeline.trainingWindows)}
          {@render asyncLane("Coordination / I/O", asyncTimeline.coordination)}
        </div>
      </div>
      <div class="tl-hint">
        Blue outline = total training time · unfilled space = preprocessing and other training work · scroll to zoom
      </div>
    {:else if layout === "timeline"}
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
      <span class="tg-tip-step">{tip.stepLabel}</span>
      <span class="tg-tip-name">{tip.label}</span>
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

  .async-timeline {
    display: flex;
    flex-direction: column;
    gap: 3px;
    min-width: 100%;
    padding-bottom: 2px;
  }

  .async-axis,
  .async-lane {
    display: grid;
    grid-template-columns: 128px minmax(0, 1fr);
    gap: 8px;
  }

  .async-axis {
    font-size: 10px;
    line-height: 14px;
    margin-bottom: 3px;
  }

  .async-axis span:last-child {
    grid-column: 2;
    justify-self: end;
  }

  .async-lane {
    align-items: center;
  }

  .async-lane-label {
    font-size: 10px;
    line-height: 14px;
    overflow: hidden;
    text-align: right;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .async-lane-track {
    position: relative;
  }

  .async-seg {
    position: absolute;
    top: 0;
    z-index: 2;
    min-width: 1px;
  }

  .async-seg.training-inner-seg {
    top: 2px;
    z-index: 3;
    height: calc(100% - 4px);
    border-radius: 1px;
  }

  .training-window {
    position: absolute;
    top: 0;
    z-index: 1;
    height: 100%;
    border: 1px solid var(--training-color);
    border-radius: 3px;
    background: color-mix(in srgb, var(--training-color) 28%, transparent);
    cursor: pointer;
  }

  .training-window.active {
    outline: 1px solid var(--text-bright);
    outline-offset: -1px;
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
    min-width: 3px;
    overflow: hidden;
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

  .bar.tl-bar.async-lane-track {
    height: 14px;
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
