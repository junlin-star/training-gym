<script>
  // Density heatmap: how a training run's advantage distribution evolves over
  // steps. Each column is one step (rollout_id); the colour at a given y encodes
  // how *concentrated* that step's advantages are around that value — bright =
  // dense (lots of samples there), dark = sparse. Read left→right: a tall bright
  // core that stays put = stable spread; a thinning/shifting core = the
  // distribution collapsing or drifting.
  //
  // The list endpoint only carries per-step quantiles (min/p10/p25/p50/p75/p90/
  // max), so we reconstruct a piecewise-uniform density: each inter-quantile
  // segment holds a known fraction of the mass over its y-range, and
  // density = mass / height. Colours are normalised across the whole run so
  // steps are comparable.

  let { steps = [] } = $props();

  const W = 640;
  const H = 200;
  const PAD = 3;

  function num(v, fallback = 0) {
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
  }

  function fmt(v) {
    return Number.isFinite(v) ? v.toFixed(3) : "—";
  }

  // viridis-ish sequential ramp (sparse → dense).
  const STOPS = [
    [0.0, [68, 1, 84]],
    [0.25, [59, 82, 139]],
    [0.5, [33, 145, 140]],
    [0.75, [94, 201, 98]],
    [1.0, [253, 231, 37]],
  ];

  function ramp(t) {
    t = Math.max(0, Math.min(1, t));
    for (let i = 1; i < STOPS.length; i++) {
      if (t <= STOPS[i][0]) {
        const [t0, c0] = STOPS[i - 1];
        const [t1, c1] = STOPS[i];
        const f = (t - t0) / (t1 - t0 || 1);
        const c = c0.map((v, k) => Math.round(v + (c1[k] - v) * f));
        return `rgb(${c[0]},${c[1]},${c[2]})`;
      }
    }
    const c = STOPS[STOPS.length - 1][1];
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  const SEGMENTS = [
    ["min", "p10", 0.1],
    ["p10", "p25", 0.15],
    ["p25", "p50", 0.25],
    ["p50", "p75", 0.25],
    ["p75", "p90", 0.15],
    ["p90", "max", 0.1],
  ];

  let model = $derived.by(() => {
    const pts = (steps || [])
      .filter((s) => s && s.stats)
      .map((s) => {
        const st = s.stats;
        const q = st.quantiles || {};
        return {
          x: num(s.rollout_id),
          min: num(st.min),
          max: num(st.max),
          p10: num(q.p10, num(st.min)),
          p25: num(q.p25, num(st.min)),
          p50: num(q.p50, num(st.mean)),
          p75: num(q.p75, num(st.max)),
          p90: num(q.p90, num(st.max)),
        };
      });
    if (pts.length < 2) return null;

    const yLo = Math.min(...pts.map((p) => p.min));
    const yHi = Math.max(...pts.map((p) => p.max));
    const ySpan = yHi - yLo || 1;
    // Floor on a segment's height so a degenerate (p_i == p_{i+1}) slice doesn't
    // produce infinite density and wash out the colour scale.
    const minH = ySpan * 0.004;
    const sy = (v) => H - ((v - yLo) / ySpan) * (H - 2 * PAD) - PAD;
    const colW = W / pts.length;

    const cells = [];
    let maxDensity = 0;
    pts.forEach((p, i) => {
      for (const [lo, hi, mass] of SEGMENTS) {
        const height = Math.max(p[hi] - p[lo], 0);
        const density = mass / Math.max(height, minH);
        if (density > maxDensity) maxDensity = density;
        cells.push({ i, loV: p[lo], hiV: p[hi], density });
      }
    });

    const rects = cells.map((c) => {
      const yTop = sy(c.hiV);
      const yBot = sy(c.loV);
      return {
        x: (c.i * colW).toFixed(2),
        w: (colW + 0.5).toFixed(2),
        y: yTop.toFixed(2),
        h: Math.max(yBot - yTop, 0.5).toFixed(2),
        // sqrt lifts the low end so sparse-but-present regions stay visible.
        fill: ramp(Math.sqrt(c.density / (maxDensity || 1))),
      };
    });

    const median = pts
      .map(
        (p, i) =>
          `${i ? "L" : "M"} ${(i * colW + colW / 2).toFixed(1)} ${sy(p.p50).toFixed(1)}`,
      )
      .join(" ");

    return {
      rects,
      median,
      yLo,
      yHi,
      zeroY: yLo <= 0 && yHi >= 0 ? sy(0) : null,
      firstX: pts[0].x,
      lastX: pts[pts.length - 1].x,
      latest: pts[pts.length - 1],
    };
  });
</script>

{#if model}
  <div class="heat-legend">
    <span>sparse</span>
    <span class="heat-bar"></span>
    <span>dense</span>
    <span class="heat-legend-sep"></span>
    <span class="heat-legend-item"><span class="heat-swatch median"></span>median</span>
  </div>
  <svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" aria-hidden="true">
    {#each model.rects as r}
      <rect x={r.x} y={r.y} width={r.w} height={r.h} fill={r.fill} />
    {/each}
    {#if model.zeroY != null}
      <line
        x1="0"
        x2={W}
        y1={model.zeroY}
        y2={model.zeroY}
        stroke="#fff"
        stroke-width="0.75"
        stroke-opacity="0.45"
        stroke-dasharray="4 4"
      />
    {/if}
    <path d={model.median} fill="none" stroke="#fff" stroke-width="1.25" stroke-opacity="0.85" />
  </svg>
  <div class="fan-meta">
    <span>min {fmt(model.yLo)}</span>
    <span>latest median {fmt(model.latest.p50)}</span>
    <span>max {fmt(model.yHi)}</span>
  </div>
  <div class="fan-axis">
    <span>step {model.firstX}</span>
    <span class="fan-axis-label">training step</span>
    <span>step {model.lastX}</span>
  </div>
{:else}
  <div class="empty">Advantage distribution needs ≥2 steps of data.</div>
{/if}

<style>
  svg {
    width: 100%;
    height: 200px;
    display: block;
    background: #0a0e14;
    border-radius: 4px;
  }
  .heat-legend {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 8px;
    font-size: 11px;
    color: var(--muted);
  }
  .heat-bar {
    width: 90px;
    height: 10px;
    border-radius: 2px;
    background: linear-gradient(
      to right,
      rgb(68, 1, 84),
      rgb(59, 82, 139),
      rgb(33, 145, 140),
      rgb(94, 201, 98),
      rgb(253, 231, 37)
    );
  }
  .heat-legend-sep {
    width: 1px;
    height: 12px;
    background: var(--border, #2a2a2a);
    margin: 0 4px;
  }
  .heat-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }
  .heat-swatch.median {
    width: 12px;
    height: 2px;
    background: #fff;
    opacity: 0.85;
  }
  .fan-meta {
    display: flex;
    justify-content: space-between;
    margin-top: 6px;
    font-size: 11px;
    color: var(--muted);
  }
  .fan-axis {
    display: flex;
    justify-content: space-between;
    margin-top: 2px;
    font-size: 10px;
    color: var(--muted);
  }
  .fan-axis-label {
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .empty {
    color: var(--muted);
    font-size: 12px;
    padding: 8px 0;
  }
</style>
