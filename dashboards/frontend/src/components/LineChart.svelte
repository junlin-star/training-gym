<script>
  let {
    title = "",
    data = [],
    height = 140,
    color = "var(--accent)",
    ariaLabel = title || "Line chart",
    formatX = (row) => String(row?.x ?? ""),
    formatY = (value) => String(value),
  } = $props();

  let chartEl = $state(null);
  let hoveredIndex = $state(null);
  let pendingEvent = null;
  let frame = null;

  let rows = $derived(
    (Array.isArray(data) ? data : [])
      .map((row, index) => ({
        ...row,
        index,
        x: Number(row?.x),
        y: Number(row?.y),
      }))
      .filter((row) => Number.isFinite(row.x) && Number.isFinite(row.y)),
  );

  let xMin = $derived(rows.length ? Math.min(...rows.map((row) => row.x)) : 0);
  let xMax = $derived(rows.length ? Math.max(...rows.map((row) => row.x)) : 1);
  let xSpan = $derived(xMax - xMin || 1);
  let yMin = $derived(rows.length ? Math.min(0, ...rows.map((row) => row.y)) : 0);
  let yMax = $derived(rows.length ? Math.max(0, ...rows.map((row) => row.y)) : 1);
  let ySpan = $derived(yMax - yMin || 1);

  function point(row) {
    const x = rows.length === 1 ? 2 : ((row.x - xMin) / xSpan) * 100;
    const y = rows.length === 1 ? 50 : 100 - ((row.y - yMin) / ySpan) * 96 - 2;
    return { x, y };
  }

  let path = $derived(
    rows
      .map((row, index) => {
        const p = point(row);
        return `${index === 0 ? "M" : "L"} ${p.x.toFixed(3)} ${p.y.toFixed(3)}`;
      })
      .join(" "),
  );
  let hoveredRow = $derived(
    hoveredIndex == null ? null : rows[Math.max(0, Math.min(rows.length - 1, hoveredIndex))],
  );
  let hoveredPoint = $derived(hoveredRow ? point(hoveredRow) : null);
  let reverseTooltip = $derived(hoveredPoint ? hoveredPoint.x > 72 : false);

  function updateHoverFromPointer(event) {
    if (!chartEl || !rows.length) return;
    const rect = chartEl.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
    const ratio = rect.width ? x / rect.width : 0;
    const targetX = xMin + ratio * xSpan;
    let nearestIndex = 0;
    let nearestDistance = Infinity;
    rows.forEach((row, index) => {
      const distance = Math.abs(row.x - targetX);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestIndex = index;
      }
    });
    hoveredIndex = nearestIndex;
  }

  function onPointerMove(event) {
    pendingEvent = event;
    if (frame != null) return;
    frame = requestAnimationFrame(() => {
      frame = null;
      if (pendingEvent) updateHoverFromPointer(pendingEvent);
    });
  }

  function onPointerLeave() {
    hoveredIndex = null;
    pendingEvent = null;
  }

  $effect(() => {
    return () => {
      if (frame != null) cancelAnimationFrame(frame);
    };
  });
</script>

<div class="chart">
  {#if title}
    <div class="chart-title">{title}</div>
  {/if}

  {#if rows.length}
    <div
      class="chart-frame"
      bind:this={chartEl}
      style:height={`${height}px`}
      role="img"
      aria-label={ariaLabel}
      onpointermove={onPointerMove}
      onpointerleave={onPointerLeave}
    >
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        {#if rows.length > 1}
          <path d={path} fill="none" stroke={color} stroke-width="1.5" vector-effect="non-scaling-stroke" />
        {:else}
          <path d={path} fill="none" stroke={color} stroke-width="1.5" vector-effect="non-scaling-stroke" />
        {/if}
        {#if hoveredPoint}
          <line
            x1={hoveredPoint.x}
            x2={hoveredPoint.x}
            y1="0"
            y2="100"
            class="hover-line"
            vector-effect="non-scaling-stroke"
          />
        {/if}
      </svg>

      {#if rows.length === 1}
        {@const p = point(rows[0])}
        <span
          class="point-dot"
          style:left={`${p.x}%`}
          style:top={`${p.y}%`}
          style:background={color}
        ></span>
      {/if}

      {#if hoveredPoint}
        <span
          class="point-dot hover-dot"
          style:left={`${hoveredPoint.x}%`}
          style:top={`${hoveredPoint.y}%`}
          style:background={color}
        ></span>
      {/if}

      {#if hoveredRow && hoveredPoint}
        <div
          class="chart-tooltip"
          class:reverse={reverseTooltip}
          style:left={`${hoveredPoint.x}%`}
        >
          <div class="tooltip-title">{formatX(hoveredRow)}</div>
          <div class="tooltip-value">{formatY(hoveredRow.y, hoveredRow)}</div>
        </div>
      {/if}
    </div>
  {:else}
    <div class="chart-empty">No data.</div>
  {/if}
</div>

<style>
  .chart {
    min-width: 0;
  }

  .chart-title {
    color: var(--text-bright);
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 6px;
  }

  .chart-frame {
    position: relative;
    background: var(--color-c-gray-08, #1c1c1c);
    border-radius: 6px;
    cursor: crosshair;
  }

  svg {
    display: block;
    width: 100%;
    height: 100%;
  }

  .hover-line {
    stroke: rgba(255, 255, 255, 0.22);
    stroke-width: 1;
  }

  .point-dot {
    position: absolute;
    z-index: 1;
    width: 7px;
    height: 7px;
    border-radius: 999px;
    pointer-events: none;
    transform: translate(-50%, -50%);
    box-shadow: 0 0 0 2px var(--color-c-gray-08, #1c1c1c);
  }

  .hover-dot {
    z-index: 2;
    width: 8px;
    height: 8px;
  }

  .chart-tooltip {
    position: absolute;
    top: 8px;
    z-index: 2;
    pointer-events: none;
    transform: translateX(12px);
    border-radius: 4px;
    background: rgba(0, 0, 0, 0.82);
    color: white;
    padding: 5px 7px;
    font-size: 12px;
    line-height: 16px;
    white-space: nowrap;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.28);
  }

  .chart-tooltip.reverse {
    transform: translateX(calc(-100% - 12px));
  }

  .tooltip-title {
    color: rgba(255, 255, 255, 0.72);
  }

  .tooltip-value {
    color: white;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }

  .chart-empty {
    color: var(--muted);
    font-size: 12px;
    line-height: 16px;
  }
</style>
