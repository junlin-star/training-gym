<script>
  import { Download, ZoomIn, ZoomOut } from "lucide-svelte";
  import { colorFor, fmtSecs, labelFor, rolloutTimeline } from "../lib/timing.js";

  let { timings = null, downloadName = "substep_timing.json" } = $props();

  // Zoom bounds: 1 = every rollout fits the width, MAX_ZOOM = deepest look at
  // one phase. Rollout columns keep their relative widths at every zoom, so a
  // wider column is a longer rollout however far in you are.
  const MIN_ZOOM = 1;
  const MAX_ZOOM = 64;
  const ZOOM_BTN_FACTOR = 1.5;
  const WHEEL_SENSITIVITY = 0.0015;

  let rollouts = $derived.by(() =>
    Object.entries(timings || {})
      .map(([id, lanes]) => ({ id: Number(id), ...rolloutTimeline(lanes) }))
      .sort((a, b) => a.id - b.id),
  );

  let measured = $derived(rollouts.filter((r) => r.rows.length > 0));
  let legend = $derived.by(() => {
    const names = new Set();
    for (const rollout of measured)
      for (const row of rollout.rows) for (const bar of row) names.add(bar.name);
    return [...names];
  });

  function downloadJson() {
    const blob = new Blob([JSON.stringify(timings || {}, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = downloadName;
    a.click();
    URL.revokeObjectURL(url);
  }

  let zoom = $state(1);
  let viewport = $state(null);

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

  let tip = $state(null);
  let pinned = $state(false);

  function isActive(rolloutId, bar) {
    return tip && tip.rolloutId === rolloutId && tip.bar.key === bar.key;
  }

  function tipFor(e, rolloutId, bar) {
    return { x: e.clientX, y: e.clientY, rolloutId, bar };
  }

  function showTip(e, rolloutId, bar) {
    if (pinned) return;
    tip = tipFor(e, rolloutId, bar);
  }

  function moveTip(e) {
    if (pinned || !tip) return;
    tip = { ...tip, x: e.clientX, y: e.clientY };
  }

  function hideTip() {
    if (pinned) return;
    tip = null;
  }

  function pinTip(e, rolloutId, bar) {
    e.stopPropagation();
    if (pinned && isActive(rolloutId, bar)) {
      pinned = false;
      tip = null;
      return;
    }
    pinned = true;
    tip = tipFor(e, rolloutId, bar);
  }

  function clearPin() {
    if (!pinned) return;
    pinned = false;
    tip = null;
  }
</script>

<svelte:window onclick={clearPin} />

<div class="run-timeline">
  {#if measured.length === 0}
    <div class="empty">No substep timing recorded for these rollouts yet.</div>
  {:else}
    <div class="toolbar">
      <div class="legend">
        {#each legend as name (name)}
          <span class="legend-item">
            <span class="swatch" style:background={colorFor(name)}></span>
            {labelFor(name)}
          </span>
        {/each}
      </div>
      <div class="controls">
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
        <button class="dl-btn" onclick={downloadJson} title="Download timing as JSON">
          <Download size={13} />
          Download JSON
        </button>
      </div>
    </div>

    <div class="viewport" bind:this={viewport} use:wheelZoom>
      <div class="track" style:width={`${zoom * 100}%`}>
        {#each rollouts as rollout (rollout.id)}
          <div class="rollout" style:flex-grow={Math.max(rollout.span, 0.001)}>
            <div class="rollout-head">
              <span class="rollout-name">Rollout {rollout.id}</span>
              <span class="rollout-span">{fmtSecs(rollout.span)}</span>
            </div>
            {#if rollout.rows.length === 0}
              <div class="row row-empty"></div>
            {:else}
              {#each rollout.rows as row, index (index)}
                <div class="row" class:nested={row[0].depth > 0}>
                  {#each row as bar (bar.key)}
                    <button
                      class="bar"
                      aria-label={`${labelFor(bar.name)} ${fmtSecs(bar.duration)}`}
                      class:banded={bar.banded}
                      class:active={pinned && isActive(rollout.id, bar)}
                      style:background={colorFor(bar.name)}
                      style:left={`${(bar.start / rollout.span) * 100}%`}
                      style:width={`${Math.max(((bar.end - bar.start) / rollout.span) * 100, 0.4)}%`}
                      onmouseenter={(e) => showTip(e, rollout.id, bar)}
                      onmousemove={moveTip}
                      onmouseleave={hideTip}
                      onclick={(e) => pinTip(e, rollout.id, bar)}
                    ></button>
                  {/each}
                </div>
              {/each}
            {/if}
          </div>
        {/each}
      </div>
    </div>
    <div class="hint">
      Hover a bar for its phase and exact times · scroll to zoom · shift-scroll or drag to
      pan · click a bar to pin its details. Bars share one clock per rollout: a thin bar
      ran inside the bar above it, and a bar on a row of its own overlapped work it is not
      part of.
    </div>
  {/if}
</div>

{#if tip}
  <div class="tg-tip" class:pinned style:left={`${tip.x}px`} style:top={`${tip.y}px`}>
    <span class="tg-tip-head">
      Rollout {tip.rolloutId}
      <!-- The driver is where a phase lives unless it says otherwise. -->
      {#if tip.bar.role !== "driver"}· measured on the {tip.bar.role}{/if}
    </span>
    <span class="tg-tip-name">{labelFor(tip.bar.name)}</span>
    <span class="tg-tip-dur">{fmtSecs(tip.bar.duration)}</span>
    <span class="tg-tip-when">
      {fmtSecs(tip.bar.start)} → {fmtSecs(tip.bar.end)} into the rollout
    </span>
    {#if tip.bar.banded}
      <span class="tg-tip-when">
        {tip.bar.count} runs · average {fmtSecs(tip.bar.average)} · longest
        {fmtSecs(tip.bar.longest)}
      </span>
      <span class="tg-tip-when">
        spread over {fmtSecs(tip.bar.end - tip.bar.start)}, not one run
      </span>
    {/if}
    {#if tip.bar.inside}
      <span class="tg-tip-when">ran inside {labelFor(tip.bar.inside)}</span>
    {/if}
    {#if tip.bar.overlaps.length}
      <span class="tg-tip-when">
        ran at the same time as {tip.bar.overlaps.map(labelFor).join(", ")}
      </span>
    {/if}
  </div>
{/if}

<style>
  .run-timeline {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .empty {
    color: var(--color-c-gray-45, #6e6e6e);
    font-size: 0.85rem;
    padding: 0.5rem 0;
  }

  .toolbar {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 14px;
    font-size: 11px;
    color: var(--muted);
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

  .controls {
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

  .dl-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
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

  .viewport {
    overflow-x: auto;
    overflow-y: hidden;
    padding-bottom: 6px;
    overscroll-behavior-x: contain;
    touch-action: pan-x;
    -webkit-overflow-scrolling: touch;
  }

  .track {
    display: flex;
    gap: 3px;
    min-width: 100%;
  }

  .rollout {
    display: flex;
    flex-direction: column;
    gap: 2px;
    flex-basis: 0;
    min-width: 2px;
  }

  .rollout-head {
    display: flex;
    align-items: baseline;
    gap: 6px;
    font-size: 10px;
    line-height: 14px;
    white-space: nowrap;
    overflow: hidden;
  }

  .rollout-name {
    color: var(--text-bright);
    font-weight: 500;
  }

  .rollout-span {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .row {
    position: relative;
    height: 16px;
    border-radius: 3px;
    background: var(--color-c-gray-08, #1c1c1c);
  }

  /* Bars that ran inside the bar above them, drawn thin so the containing
     phase stays the one you read first. */
  .row.nested {
    height: 7px;
    background: none;
  }

  .row-empty {
    background: var(--color-c-gray-10, #2f2f2f);
  }

  .bar {
    position: absolute;
    top: 0;
    height: 100%;
    min-width: 2px;
    padding: 0 3px;
    border: none;
    border-radius: 3px;
    overflow: hidden;
    cursor: pointer;
    transition: filter 0.1s ease;
  }

  .bar:hover {
    filter: brightness(1.25);
  }

  .bar.active {
    outline: 1px solid var(--text-bright, #fff);
    outline-offset: -1px;
    filter: brightness(1.3);
  }

  /* A phase that ran too many times to keep each run: one band over the span
     it covered, hatched so it doesn't read as a single continuous run. */
  .bar.banded {
    background-image: repeating-linear-gradient(
      45deg,
      rgba(0, 0, 0, 0.28),
      rgba(0, 0, 0, 0.28) 3px,
      transparent 3px,
      transparent 6px
    );
  }

  .hint {
    font-size: 10px;
    color: var(--muted);
    opacity: 0.7;
  }

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

  .tg-tip-head {
    color: var(--muted);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .tg-tip-name {
    color: var(--text-bright, #fff);
    font-weight: 600;
  }

  .tg-tip-dur,
  .tg-tip-when {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }
</style>
