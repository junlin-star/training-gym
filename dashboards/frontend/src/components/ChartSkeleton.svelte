<script>
  // Loading placeholder for the graph components on the run detail page. Mirrors
  // the footprint of the real chart (optional title, a plot box the same height,
  // and mark shapes appropriate to the chart type) so the layout doesn't jump
  // when data arrives.
  //
  //   variant   "line" | "bars" | "violins"  — which chart it stands in for
  //   height    number (px plot height)
  //   count     number of marks (bars / violins)
  //   showTitle render a title pulse above the plot (charts whose title is drawn
  //             by the chart component itself, e.g. the reward LineChart)

  let {
    variant = "bars",
    height = 150,
    count = 12,
    showTitle = false,
  } = $props();

  // Deterministic mark heights (no Math.random — keeps SSR/hydration stable and
  // avoids the reroll-on-every-render flicker). A blend of two sines reads as an
  // irregular distribution rather than an obvious repeating wave.
  let marks = $derived(
    Array.from({ length: Math.max(1, count) }, (_, i) => {
      const t = Math.abs(Math.sin(i * 1.3) * 0.6 + Math.sin(i * 0.7 + 1) * 0.4);
      return 22 + Math.round(t * 66); // 22%–88% of plot height
    }),
  );
</script>

<div class="cs" aria-hidden="true">
  {#if showTitle}
    <span class="cs-pulse cs-title"></span>
  {/if}
  <div class="cs-plot" class:bordered={variant !== "line"} style:height={`${height}px`}>
    {#if variant === "line"}
      <span class="cs-grid" style:top="25%"></span>
      <span class="cs-grid" style:top="50%"></span>
      <span class="cs-grid" style:top="75%"></span>
      <span class="cs-pulse cs-area"></span>
    {:else}
      <div class="cs-marks">
        {#each marks as h, i (i)}
          <span
            class="cs-pulse"
            class:cs-bar={variant === "bars"}
            class:cs-violin={variant === "violins"}
            style:height={`${h}%`}
          ></span>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .cs {
    width: 100%;
  }
  .cs-pulse {
    display: block;
    background: #2f2f2f;
    border-radius: 6px;
    animation: chart-skeleton-pulse 1.2s ease-in-out infinite;
  }
  .cs-title {
    width: 40%;
    max-width: 160px;
    height: 12px;
    margin-bottom: 8px;
  }
  .cs-plot {
    position: relative;
    width: 100%;
  }
  .cs-plot.bordered {
    border-bottom: 1px solid var(--border, #2f2f2f);
  }

  /* line / area */
  .cs-grid {
    position: absolute;
    left: 0;
    right: 0;
    height: 1px;
    background: color-mix(in srgb, var(--muted, #a3a3a3) 18%, transparent);
  }
  .cs-area {
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 58%;
    border-radius: 8px 8px 0 0;
    opacity: 0.5;
  }

  /* bars / violins */
  .cs-marks {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 3px;
  }
  .cs-bar {
    flex: 1 1 0;
    min-width: 0;
    border-radius: 3px 3px 0 0;
  }
  .cs-violin {
    flex: 1 1 0;
    min-width: 0;
    max-width: 22px;
    margin: auto 0;
    border-radius: 999px;
    opacity: 0.65;
  }

  /* Stagger the marks so the row shimmers rather than blinking in unison. */
  .cs-marks .cs-pulse:nth-child(3n + 2) {
    animation-delay: 0.2s;
  }
  .cs-marks .cs-pulse:nth-child(3n) {
    animation-delay: 0.4s;
  }

  @keyframes chart-skeleton-pulse {
    0%,
    100% {
      opacity: 1;
    }
    50% {
      opacity: 0.45;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .cs-pulse {
      animation: none;
      opacity: 0.7;
    }
  }
</style>
