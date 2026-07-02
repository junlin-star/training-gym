<script>
  // Generic grouped ("comparative") bar chart: one cluster per category, one bar
  // per series within the cluster, so several series can be compared side-by-side
  // at each category. Hovering a bar (or its cluster) surfaces a tooltip with the
  // category and every series' value; each bar also carries a native `title` as a
  // no-JS / screen-reader fallback.
  //
  // Signed values are supported: bars grow up from a shared zero baseline for
  // positive values and down for negative ones.
  //
  // Props
  //   categories        string[]                       — one cluster per entry (x labels)
  //   series            { name, values:number[], color? }[]
  //   height            number (px plot height)
  //   format            (value) => string              — tooltip / label formatter
  //   axisLabel         string                         — caption under the x-axis
  //   showLegend        boolean (auto-off for 1 series)
  //   showCategoryLabels boolean                       — per-cluster x labels
  //   showValueLabels   boolean                        — direct labels above bars
  //   yMax              number | null                  — fix the positive top of the scale
  //   emptyText         string

  let {
    categories = [],
    series = [],
    height = 150,
    format = (v) => `${v}`,
    axisLabel = "",
    showLegend = true,
    showCategoryLabels = true,
    showValueLabels = false,
    yMax = null,
    emptyText = "No data to compare.",
  } = $props();

  // Validated dark-surface categorical ramp (skill reference palette), in fixed
  // order — never cycled cosmetically; the order itself is the CVD-safety choice.
  const DEFAULT_PALETTE = [
    "#3987e5", // blue
    "#199e70", // aqua
    "#c98500", // yellow
    "#008300", // green
    "#9085e9", // violet
    "#e66767", // red
    "#d55181", // magenta
    "#d95926", // orange
  ];

  const colorFor = (s, si) => s.color || DEFAULT_PALETTE[si % DEFAULT_PALETTE.length];

  function finite(v) {
    const n = Number(v);
    return Number.isFinite(n) ? n : 0;
  }

  let model = $derived.by(() => {
    const cats = categories || [];
    const ser = (series || []).filter((s) => s && Array.isArray(s.values));
    if (!cats.length || !ser.length) return null;

    // Loop (not Math.min(...arr)): value arrays can be large enough that the
    // spread would blow the engine's max-argument limit.
    let dataMax = 0;
    let dataMin = 0;
    for (const s of ser) {
      for (const raw of s.values) {
        const v = finite(raw);
        if (v > dataMax) dataMax = v;
        if (v < dataMin) dataMin = v;
      }
    }
    const posTop = yMax != null && Number.isFinite(yMax) ? Math.max(yMax, dataMax) : dataMax;
    const span = posTop - dataMin || 1;
    const zeroPct = (posTop / span) * 100; // distance from top down to the zero line

    const clusters = cats.map((label, ci) => {
      const bars = ser.map((s, si) => {
        const v = finite(s.values[ci]);
        const magPct = (Math.abs(v) / span) * 100;
        return {
          si,
          name: s.name ?? `series ${si + 1}`,
          color: colorFor(s, si),
          value: v,
          positive: v >= 0,
          // Keep a sliver visible for tiny-but-nonzero values.
          heightPct: v === 0 ? 0 : Math.max(magPct, 0.8),
          topPct: v >= 0 ? zeroPct - Math.max(magPct, v === 0 ? 0 : 0.8) : zeroPct,
        };
      });
      return { label: label == null ? `${ci}` : String(label), ci, bars };
    });

    return {
      clusters,
      series: ser.map((s, si) => ({ name: s.name ?? `series ${si + 1}`, color: colorFor(s, si) })),
      zeroPct,
      hasNegative: dataMin < 0,
      multi: ser.length > 1,
    };
  });

  let hover = $state(null); // { ci, si }
  let ptr = $state({ x: 0, y: 0, w: 0, h: 0 });

  function onPlotMove(e) {
    const r = e.currentTarget.getBoundingClientRect();
    ptr = { x: e.clientX - r.left, y: e.clientY - r.top, w: r.width, h: r.height };
  }

  function enterBar(ci, si) {
    hover = { ci, si };
  }

  function clearHover() {
    hover = null;
  }

  let tip = $derived.by(() => {
    if (!hover || !model) return null;
    const cluster = model.clusters[hover.ci];
    if (!cluster) return null;
    const TIP_W = 168;
    const left = Math.max(0, Math.min(ptr.x + 12, Math.max(0, ptr.w - TIP_W)));
    const top = Math.max(0, Math.min(ptr.y, ptr.h));
    return {
      left,
      top,
      label: cluster.label,
      rows: cluster.bars.map((b) => ({
        name: b.name,
        color: b.color,
        value: format(b.value),
        active: b.si === hover.si,
      })),
    };
  });

  const barTitle = (clusterLabel, bar) =>
    `${clusterLabel} · ${bar.name}: ${format(bar.value)}`;
</script>

{#if model}
  {#if showLegend && model.multi}
    <div class="cbc-legend">
      {#each model.series as s (s.name)}
        <span class="cbc-legend-item">
          <span class="cbc-swatch" style:background={s.color}></span>
          {s.name}
        </span>
      {/each}
    </div>
  {/if}

  <div
    class="cbc-plot"
    class:has-negative={model.hasNegative}
    style:height={`${height}px`}
    role="img"
    aria-label={`Comparative bar chart across ${model.clusters.length} categories`}
    onpointermove={onPlotMove}
    onpointerleave={clearHover}
  >
    <div class="cbc-baseline" style:top={`${model.zeroPct}%`}></div>

    <div class="cbc-clusters">
      {#each model.clusters as cluster (cluster.ci)}
        <div class="cbc-cluster" class:dim={hover && hover.ci !== cluster.ci}>
          {#each cluster.bars as bar (bar.si)}
            <div class="cbc-slot">
              <div
                class="cbc-bar"
                class:negative={!bar.positive}
                class:active={hover && hover.ci === cluster.ci && hover.si === bar.si}
                style:background={bar.color}
                style:top={`${bar.topPct}%`}
                style:height={`${bar.heightPct}%`}
                title={barTitle(cluster.label, bar)}
                role="presentation"
                onpointerenter={() => enterBar(cluster.ci, bar.si)}
              >
                {#if showValueLabels && bar.value !== 0}
                  <span class="cbc-value-label">{format(bar.value)}</span>
                {/if}
              </div>
            </div>
          {/each}
        </div>
      {/each}
    </div>

    {#if tip}
      <div class="cbc-tooltip" style:left={`${tip.left}px`} style:top={`${tip.top}px`}>
        <div class="cbc-tip-label">{tip.label}</div>
        {#each tip.rows as row (row.name)}
          <div class="cbc-tip-row" class:active={row.active}>
            <span class="cbc-swatch" style:background={row.color}></span>
            <span class="cbc-tip-name">{row.name}</span>
            <span class="cbc-tip-value">{row.value}</span>
          </div>
        {/each}
      </div>
    {/if}
  </div>

  {#if showCategoryLabels}
    <div class="cbc-xaxis">
      {#each model.clusters as cluster (cluster.ci)}
        <span class="cbc-xlabel" title={cluster.label}>{cluster.label}</span>
      {/each}
    </div>
  {/if}
  {#if axisLabel}
    <div class="cbc-axis-caption">{axisLabel}</div>
  {/if}
{:else}
  <div class="cbc-empty">{emptyText}</div>
{/if}

<style>
  .cbc-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin-bottom: 8px;
    font-size: 11px;
    color: var(--muted);
  }
  .cbc-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .cbc-swatch {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    flex: 0 0 auto;
  }

  .cbc-plot {
    position: relative;
    width: 100%;
    border-bottom: 1px solid var(--border, #2f2f2f);
  }
  /* When the data straddles zero, the baseline floats inside the plot, so give
     it its own visible rule instead of leaning on the bottom border. */
  .cbc-baseline {
    position: absolute;
    left: 0;
    right: 0;
    height: 0;
    border-top: 1px dashed color-mix(in srgb, var(--muted) 50%, transparent);
    opacity: 0;
    pointer-events: none;
  }
  .cbc-plot.has-negative .cbc-baseline {
    opacity: 1;
  }

  .cbc-clusters {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: stretch;
    gap: 3px;
  }
  .cbc-cluster {
    position: relative;
    flex: 1 1 0;
    display: flex;
    gap: 2px;
    min-width: 0;
    transition: opacity 0.1s ease;
  }
  .cbc-cluster.dim {
    opacity: 0.45;
  }
  .cbc-slot {
    position: relative;
    flex: 1 1 0;
    min-width: 0;
  }
  .cbc-bar {
    position: absolute;
    left: 0;
    right: 0;
    min-height: 1px;
    border-radius: 3px 3px 0 0;
    transition:
      filter 0.1s ease,
      box-shadow 0.1s ease;
  }
  .cbc-bar.negative {
    border-radius: 0 0 3px 3px;
  }
  .cbc-bar.active {
    filter: brightness(1.15);
    box-shadow: 0 0 0 2px var(--surface, #2f2f2f);
  }
  .cbc-value-label {
    position: absolute;
    top: -14px;
    left: 50%;
    transform: translateX(-50%);
    font-size: 9px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
    pointer-events: none;
  }

  .cbc-tooltip {
    position: absolute;
    z-index: 5;
    transform: translateY(-50%);
    min-width: 120px;
    max-width: 168px;
    padding: 6px 8px;
    border-radius: 6px;
    background: rgba(10, 14, 20, 0.94);
    border: 1px solid var(--border-strong, #464646);
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    pointer-events: none;
    font-size: 11px;
  }
  .cbc-tip-label {
    color: var(--text-bright, #fff);
    font-weight: 600;
    margin-bottom: 4px;
    font-variant-numeric: tabular-nums;
  }
  .cbc-tip-row {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--muted);
    line-height: 16px;
  }
  .cbc-tip-row.active {
    color: var(--text-bright, #fff);
  }
  .cbc-tip-name {
    flex: 1 1 auto;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .cbc-tip-value {
    font-variant-numeric: tabular-nums;
  }

  .cbc-xaxis {
    display: flex;
    gap: 3px;
    margin-top: 4px;
  }
  .cbc-xlabel {
    flex: 1 1 0;
    min-width: 0;
    text-align: center;
    font-size: 10px;
    color: var(--muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    padding: 0 1px;
  }
  .cbc-axis-caption {
    margin-top: 2px;
    text-align: center;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
  }
  .cbc-empty {
    color: var(--muted);
    font-size: 12px;
    padding: 8px 0;
  }
</style>
