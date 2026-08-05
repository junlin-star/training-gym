<script>
  import { ChevronDown, ChevronRight, Download, ZoomIn, ZoomOut } from "lucide-svelte";
  import {
    CATEGORIES,
    categoryOf,
    colorFor,
    fmtSecs,
    labelFor,
    roleLabel,
    runTimeline,
  } from "../lib/timing.js";

  let { timings = null, downloadName = "substep_timing.json" } = $props();

  const MIN_ZOOM = 1;
  const MAX_ZOOM = 64;
  const ZOOM_BTN_FACTOR = 1.5;
  const WHEEL_SENSITIVITY = 0.0015;

  const ROW_HEIGHT_PX = 16;
  const ROW_GAP_PX = 3;
  const HEADER_PX = 15;
  const GROUP_GAP_PX = 10;
  const STEP_LABEL_PX = 18;

  // What each wait is actually waiting for, in steps rather than futures.
  const WAITS_ON = {
    generate_rollouts: () => "the engines generating this step's samples",
    wait_for_rollout: (id) =>
      id > 0
        ? `this step's samples, generated during step ${id - 1}`
        : "this step's samples, generation started before the loop",
    wait_for_next_rollout: (id) =>
      `step ${id + 1}'s samples, drained before the weights change`,
    evaluate_rollouts: () => "the engines running eval",
    evaluate_rollouts_end: () => "the engines running eval",
  };

  let timeline = $derived(runTimeline(timings));
  let groups = $derived(
    timeline.groups.map((group) => ({
      ...group,
      height: HEADER_PX + group.rows.length * (ROW_HEIGHT_PX + ROW_GAP_PX),
    })),
  );
  let showDetails = $state(false);
  let visibleGroups = $derived(
    groups.map((group) => {
      const rows = showDetails ? group.rows : group.rows.filter((row) => row.depth === 0);
      return {
        ...group,
        rows,
        height: HEADER_PX + rows.length * (ROW_HEIGHT_PX + ROW_GAP_PX),
      };
    }),
  );
  let trackHeight = $derived(
    visibleGroups.reduce((total, group) => total + group.height + GROUP_GAP_PX, 0),
  );

  const pct = (seconds) => (seconds / timeline.span) * 100;

  // Deeper frames are lighter, so a nested phase reads as part of its parent
  // rather than as a different kind of work.
  function fill(bar) {
    const base = colorFor(bar.name);
    if (bar.depth === 0) return base;
    return `color-mix(in srgb, ${base} ${Math.max(100 - bar.depth * 28, 40)}%, var(--color-c-gray-02, #111))`;
  }

  let zoom = $state(1);
  let viewport = $state(null);
  let viewportWidth = $state(900);

  let contentWidth = $derived(viewportWidth * zoom);
  const widthPx = (bar) => (bar.duration / timeline.span) * contentWidth;

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
    if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
    e.preventDefault();
    setZoom(zoom * Math.exp(-e.deltaY * WHEEL_SENSITIVITY), e.clientX);
  }

  function wheelZoom(node) {
    node.addEventListener("wheel", handleWheel, { passive: false });
    return {
      destroy() {
        node.removeEventListener("wheel", handleWheel);
      },
    };
  }

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

  let tip = $state(null);
  let pinned = $state(false);

  const isActive = (bar) => tip && tip.bar.key === bar.key;

  function showTip(e, bar) {
    if (pinned) return;
    tip = { x: e.clientX, y: e.clientY, bar };
  }

  function moveTip(e) {
    if (pinned || !tip) return;
    tip = { ...tip, x: e.clientX, y: e.clientY };
  }

  function hideTip() {
    if (pinned) return;
    tip = null;
  }

  function pinTip(e, bar) {
    e.stopPropagation();
    if (pinned && isActive(bar)) {
      pinned = false;
      tip = null;
      return;
    }
    pinned = true;
    tip = { x: e.clientX, y: e.clientY, bar };
  }

  function clearPin() {
    if (!pinned) return;
    pinned = false;
    tip = null;
  }
</script>

<svelte:window onclick={clearPin} />

<div class="run-timeline">
  {#if !groups.length}
    <div class="empty">No substep timing recorded for these rollouts yet.</div>
  {:else}
    <div class="toolbar">
      <div class="legend">
        {#each timeline.categories as key (key)}
          <span class="legend-item">
            <span class="swatch" style:background={CATEGORIES[key].color}></span>
            {CATEGORIES[key].label}
          </span>
        {/each}
        <span class="legend-item">
          <span class="swatch swatch-stall"></span>
          stalled, waiting on somebody else
        </span>
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
        <button
          class="dl-btn"
          onclick={() => (showDetails = !showDetails)}
          aria-pressed={showDetails}
          title={showDetails ? "Hide phase details" : "Show phase details"}
        >
          {#if showDetails}
            <ChevronDown size={13} />
            Hide phase details
          {:else}
            <ChevronRight size={13} />
            Show phase details
          {/if}
        </button>
      </div>
    </div>

    <div class="chart">
      <div class="gutter" style:padding-top={`${ROW_HEIGHT_PX + 6}px`}>
        {#each visibleGroups as group (group.key)}
          <div style:margin-bottom={`${GROUP_GAP_PX}px`}>
            <div
              class="gutter-head"
              style:height={`${HEADER_PX}px`}
              title={`${group.label} — ${group.hint}`}
            >
              {group.label}
            </div>
            {#each group.rows as row, index (index)}
              <div
                class="gutter-row"
                class:nested={row.depth > 0}
                style:height={`${ROW_HEIGHT_PX}px`}
                style:margin-bottom={`${ROW_GAP_PX}px`}
                style:padding-left={`${row.depth * 7}px`}
                title={row.label}
              >
                {row.label}
              </div>
            {/each}
          </div>
        {/each}
      </div>

      <div
        class="viewport"
        bind:this={viewport}
        bind:clientWidth={viewportWidth}
        use:wheelZoom
      >
        <div class="track" style:width={`${zoom * 100}%`}>
          <div class="steps" style:height={`${ROW_HEIGHT_PX}px`}>
            {#each timeline.steps as step (step.id)}
              <div
                class="step"
                style:left={`${pct(step.offset)}%`}
                style:width={`${Math.max(pct(step.duration), 0.05)}%`}
                title={`Step ${step.id}: ${fmtSecs(step.duration)} wall clock — ${fmtSecs(step.work)} working, ${fmtSecs(step.stalled)} waiting on the engines`}
              >
                <span class="step-text"
                  >Step {step.id} · {fmtSecs(step.duration)}</span
                >
              </div>
            {/each}
          </div>

          <div class="groups" style:height={`${trackHeight}px`}>
            {#each visibleGroups as group (group.key)}
              <div
                class="group"
                style:height={`${group.height}px`}
                style:margin-bottom={`${GROUP_GAP_PX}px`}
              >
                {#each group.rows as row, index (index)}
                  <div
                    class="row"
                    style:top={`${HEADER_PX + index * (ROW_HEIGHT_PX + ROW_GAP_PX)}px`}
                    style:height={`${ROW_HEIGHT_PX}px`}
                  >
                    {#each row.spans as bar (bar.key)}
                      <button
                        class="bar"
                        class:stall={bar.kind === "stall"}
                        class:untracked={bar.kind === "untracked"}
                        class:sampled={bar.kind === "sampled"}
                        class:active={pinned && isActive(bar)}
                        aria-label={`${labelFor(bar.name)} ${fmtSecs(bar.duration)}`}
                        style:left={`${pct(bar.offset)}%`}
                        style:width={`max(2px, ${Math.max(pct(bar.duration), 0.02)}%)`}
                        style:--bar-color={colorFor(bar.name)}
                        style:background={bar.kind === "work" ? fill(bar) : undefined}
                        onmouseenter={(e) => showTip(e, bar)}
                        onmousemove={moveTip}
                        onmouseleave={hideTip}
                        onclick={(e) => pinTip(e, bar)}
                      >
                        {#if bar.kind === "sampled"}
                          <span
                            class="tick"
                            style:left={`${Math.min((bar.average / bar.duration) * 100, 100)}%`}
                          ></span>
                        {/if}
                        {#if bar.rolloutId != null && widthPx(bar) > STEP_LABEL_PX}
                          <span class="bar-text" class:on-dark={bar.kind === "work"}>
                            {bar.rolloutId}
                          </span>
                        {/if}
                      </button>
                    {/each}
                  </div>
                {/each}
              </div>
            {/each}
          </div>
        </div>
      </div>
    </div>

    <div class="hint">
      One clock for the whole run. The training loop is on top and the machines it
      waits on are underneath, indented where one phase ran inside another. A thin
      grey line is the loop stalled on somebody else — the work itself is on the row
      below. Hatched is wall clock no phase measured{#if timeline.untracked > 0.25},
        {fmtSecs(timeline.untracked)} of it in this run{/if}. Hover for exact times,
      click to pin, scroll to zoom.
    </div>
  {/if}
</div>

{#if tip}
  <div class="tg-tip" class:pinned style:left={`${tip.x}px`} style:top={`${tip.y}px`}>
    <span class="tg-tip-head">
      {#if tip.bar.rolloutId != null}Step {tip.bar.rolloutId} ·{/if}
      {roleLabel(tip.bar.role)}
      {#if tip.bar.kind !== "untracked"}
        · {CATEGORIES[categoryOf(tip.bar.name)].label.toLowerCase()}{/if}
    </span>
    <span class="tg-tip-name">
      {tip.bar.kind === "untracked" ? "Untracked wall clock" : labelFor(tip.bar.name)}
    </span>
    <span class="tg-tip-dur">{fmtSecs(tip.bar.duration)}</span>
    <span class="tg-tip-when">
      {fmtSecs(tip.bar.offset)} → {fmtSecs(tip.bar.offset + tip.bar.duration)} into the
      run
    </span>
    {#if tip.bar.kind === "sampled"}
      <span class="tg-tip-when">
        {tip.bar.count} calls · average {fmtSecs(tip.bar.average)} · longest {fmtSecs(
          tip.bar.longest,
        )}, spread over the span rather than one run
      </span>
    {:else if tip.bar.count > 1}
      <span class="tg-tip-when">
        {tip.bar.count} runs · {fmtSecs(tip.bar.total)} of work · longest {fmtSecs(
          tip.bar.longest,
        )}
      </span>
    {/if}
    {#if tip.bar.kind === "stall"}
      <span class="tg-tip-when">
        stalled on {WAITS_ON[tip.bar.name]?.(tip.bar.rolloutId) ?? "another worker"}
      </span>
    {/if}
    {#if tip.bar.kind === "untracked"}
      <span class="tg-tip-when">
        the loop was between measured phases here — nothing claims this time
      </span>
    {/if}
    {#if tip.bar.inside}
      <span class="tg-tip-when">ran inside {labelFor(tip.bar.inside).toLowerCase()}</span>
    {/if}
  </div>
{/if}

<style>
  .run-timeline {
    display: flex;
    flex-direction: column;
    gap: 12px;
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

  .swatch-stall {
    height: 2px;
    background: var(--color-c-gray-30, #6a6a6a);
    border-radius: 1px;
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

  .chart {
    display: flex;
    align-items: flex-start;
    gap: 10px;
  }

  .gutter {
    flex-shrink: 0;
    width: 108px;
  }

  .gutter-head {
    font-size: 10px;
    line-height: 15px;
    color: var(--text-bright, #fff);
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .gutter-row {
    font-size: 9px;
    line-height: 16px;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .gutter-row.nested {
    opacity: 0.65;
  }

  .viewport {
    flex: 1;
    min-width: 0;
    overflow-x: auto;
    overflow-y: hidden;
    padding-bottom: 10px;
    scrollbar-width: thin;
    scrollbar-color: var(--color-c-gray-20, #464646) transparent;
    overscroll-behavior-x: contain;
    touch-action: pan-x;
  }

  .track {
    position: relative;
    min-width: 100%;
  }

  .steps {
    position: relative;
    margin-bottom: 6px;
  }

  .step {
    position: absolute;
    top: 0;
    bottom: 0;
    border-left: 1px solid var(--border-strong, #4a4a4a);
    background: var(--color-c-gray-05, #171717);
    overflow: hidden;
  }

  .step-text {
    display: block;
    padding: 0 4px;
    font-size: 10px;
    line-height: 16px;
    color: var(--muted);
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }

  .groups {
    position: relative;
  }

  .group {
    position: relative;
    border-top: 1px solid var(--border, #2f2f2f);
  }

  .row {
    position: absolute;
    left: 0;
    right: 0;
    pointer-events: none;
  }

  .bar {
    position: absolute;
    top: 0;
    height: 100%;
    min-width: 2px;
    padding: 0;
    border: none;
    border-radius: min(2px, 12%);
    overflow: hidden;
    cursor: pointer;
    pointer-events: auto;
    background: transparent;
    transition: filter 0.1s ease;
  }

  /* A stall is the loop doing nothing, so it is a line rather than a block: the
     work being waited on is drawn on the row underneath. */
  .bar.stall {
    background: linear-gradient(
      var(--color-c-gray-30, #6a6a6a),
      var(--color-c-gray-30, #6a6a6a)
    );
    background-size: 100% 2px;
    background-position: center;
    background-repeat: no-repeat;
  }

  .bar.untracked {
    background: repeating-linear-gradient(
      -45deg,
      var(--color-c-gray-12, #262626) 0 4px,
      transparent 4px 8px
    );
    border: 1px dashed var(--color-c-gray-25, #555);
  }

  .bar.sampled {
    background: color-mix(in srgb, var(--bar-color) 20%, transparent);
    border: 1px solid var(--bar-color);
  }

  .tick {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 1px;
    background: var(--bar-color);
  }

  .bar-text {
    position: relative;
    display: block;
    padding: 0 3px;
    font-size: 9px;
    line-height: 16px;
    color: var(--muted);
    white-space: nowrap;
    pointer-events: none;
  }

  .bar-text.on-dark {
    color: var(--color-c-gray-9, #0b0b0b);
  }

  .bar:hover {
    filter: brightness(1.25);
  }

  .bar.active {
    outline: 1px solid var(--text-bright, #fff);
    outline-offset: -1px;
    filter: brightness(1.3);
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
