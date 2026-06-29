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

  // Plot insets (in viewBox units == % of the chart box) reserved for the
  // y-axis labels on the left and the x-axis labels along the bottom.
  const ML = 9;
  const MB = 12;
  const PLOT_W = 100 - ML;
  const PLOT_H = 100 - MB;

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
  // Pad the y-range ~10% beyond the data on each side, like W&B.
  let yLo = $derived(rows.length ? Math.min(...rows.map((row) => row.y)) : 0);
  let yHi = $derived(rows.length ? Math.max(...rows.map((row) => row.y)) : 1);
  let yPad = $derived((yHi - yLo) * 0.1 || Math.abs(yHi) * 0.1 || 1);
  let yMin = $derived(yLo - yPad);
  let yMax = $derived(yHi + yPad);
  let ySpan = $derived(yMax - yMin || 1);

  let xScale = (v) => ML + ((v - xMin) / xSpan) * PLOT_W;
  let yScale = (v) => ((yMax - v) / ySpan) * PLOT_H;

  // Horizontal gridlines on nice round values (1/2/5 × 10ⁿ).
  let yTicks = $derived.by(() => {
    const rough = ySpan / 4;
    const pow = 10 ** Math.floor(Math.log10(rough));
    const norm = rough / pow;
    const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * pow;
    const out = [];
    for (let v = Math.ceil(yMin / step) * step; v <= yMax; v += step) {
      const value = Number(v.toFixed(10));
      out.push({ value, pct: yScale(value) });
    }
    return out;
  });

  let xTicks = $derived.by(() => {
    if (!rows.length) return [];
    const stride = Math.ceil(rows.length / 6);
    return rows
      .filter((_, i) => i % stride === 0 || i === rows.length - 1)
      .map((row) => ({ value: row.x, pct: xScale(row.x) }));
  });

  function point(row) {
    const x = rows.length === 1 ? ML + 2 : xScale(row.x);
    const y = yScale(row.y);
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
    // Map pointer x (0..1 of the box) back into data space, undoing the left inset.
    const targetX = xMin + Math.max(0, (ratio * 100 - ML) / PLOT_W) * xSpan;
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

<div class="min-w-0">
  {#if title}
    <div class="text-(--text-bright) text-[12px] font-[600] mb-[6px]">{title}</div>
  {/if}

  {#if rows.length}
    <div
      class="relative bg-(--color-c-gray-08,#1c1c1c) rounded-[6px] cursor-crosshair"
      bind:this={chartEl}
      style:height={`${height}px`}
      role="img"
      aria-label={ariaLabel}
      onpointermove={onPointerMove}
      onpointerleave={onPointerLeave}
    >
      <svg class="block w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        {#each yTicks as tick (tick.value)}
          <line
            x1={ML}
            x2="100"
            y1={tick.pct}
            y2={tick.pct}
            class="stroke-[rgba(255,255,255,0.07)] [stroke-width:1]"
            vector-effect="non-scaling-stroke"
          />
        {/each}
        <path d={path} fill="none" stroke={color} stroke-width="1.5" vector-effect="non-scaling-stroke" />
        {#if hoveredPoint}
          <line
            x1={hoveredPoint.x}
            x2={hoveredPoint.x}
            y1="0"
            y2={PLOT_H}
            class="stroke-[rgba(255,255,255,0.22)] [stroke-width:1]"
            vector-effect="non-scaling-stroke"
          />
        {/if}
      </svg>

      <!-- Y-axis labels, overlaid in the reserved left strip. -->
      {#each yTicks as tick (tick.value)}
        <span
          class="absolute left-0 -translate-y-1/2 text-right pr-[4px] text-[10px] leading-none text-(--muted) [font-variant-numeric:tabular-nums] pointer-events-none"
          style:top={`${tick.pct}%`}
          style:width={`${ML}%`}
        >{formatY(tick.value)}</span>
      {/each}

      <!-- X-axis labels, overlaid in the reserved bottom strip. -->
      {#each xTicks as tick (tick.value)}
        <span
          class="absolute -translate-x-1/2 text-[10px] leading-none text-(--muted) [font-variant-numeric:tabular-nums] pointer-events-none whitespace-nowrap"
          style:left={`${tick.pct}%`}
          style:top={`${PLOT_H + 3}%`}
        >{tick.value}</span>
      {/each}

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
          class="point-dot z-[2]! w-[8px]! h-[8px]!"
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
          <div class="text-[rgba(255,255,255,0.72)]">{formatX(hoveredRow)}</div>
          <div class="[color:white] font-[600] [font-variant-numeric:tabular-nums]">{formatY(hoveredRow.y, hoveredRow)}</div>
        </div>
      {/if}
    </div>
  {:else}
    <div class="text-(--muted) text-[12px] leading-[16px]">No data.</div>
  {/if}
</div>
