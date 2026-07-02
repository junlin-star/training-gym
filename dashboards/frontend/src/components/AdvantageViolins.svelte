<script>
  // A series of violin plots: one violin per step (rollout_id), so the shape of
  // the advantage distribution at each step is legible side-by-side. Used for
  // the full run (every step, read left→right as "distribution over time").
  //
  // Each violin is built as a histogram of horizontal ("rotated") bars: the
  // value axis runs vertically and every bucket is a bar whose length encodes
  // how much mass falls in that value range, mirrored around the centre so the
  // stack of bars reads as a violin. Bars are individually hoverable (SVG
  // <title>) so you can inspect a bucket's range and its share of samples.
  //
  // The list endpoint only carries per-step quantiles (min/p10/p25/p50/p75/p90/
  // max), so we reconstruct a piecewise-uniform density from them and integrate
  // it over each bucket to get that bucket's mass. Bar lengths are normalised
  // across all buckets of all violins so widths are comparable between steps.

  let { steps = [], labels = null } = $props();

  const W = 640;
  const H = 210;
  const PAD = 8;
  const N_BUCKETS = 26;
  const BAR_GAP = 0.6; // vertical gap (viewBox units) between stacked buckets

  function num(v, fallback = 0) {
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
  }

  function fmt(v) {
    return Number.isFinite(v) ? v.toFixed(3) : "—";
  }

  // Inter-quantile segments and the fraction of samples each holds.
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
          mean: num(st.mean),
          p10: num(q.p10, num(st.min)),
          p25: num(q.p25, num(st.min)),
          p50: num(q.p50, num(st.mean)),
          p75: num(q.p75, num(st.max)),
          p90: num(q.p90, num(st.max)),
        };
      });
    if (!pts.length) return null;

    const yLo = Math.min(...pts.map((p) => p.min));
    const yHi = Math.max(...pts.map((p) => p.max));
    const ySpan = yHi - yLo || 1;
    const bw = ySpan / N_BUCKETS;
    const tiny = ySpan * 1e-6;
    const sy = (v) => H - PAD - ((v - yLo) / ySpan) * (H - 2 * PAD);

    // Reconstructed piecewise density → mass falling in bucket [a, b).
    const bucketMass = (segs, a, b) => {
      let m = 0;
      for (const s of segs) {
        const len = s.hi - s.lo;
        if (len <= tiny) {
          const mid = (s.lo + s.hi) / 2;
          if (mid >= a && mid < b) m += s.mass;
        } else {
          const ov = Math.min(s.hi, b) - Math.max(s.lo, a);
          if (ov > 0) m += s.mass * (ov / len);
        }
      }
      return m;
    };

    let maxMass = 0;
    const perStep = pts.map((p) => {
      const segs = SEGMENTS.map(([lo, hi, mass]) => ({
        lo: p[lo],
        hi: p[hi],
        mass,
      }));
      const buckets = Array.from({ length: N_BUCKETS }, (_, j) => {
        const lo = yLo + j * bw;
        const hi = j === N_BUCKETS - 1 ? yHi + tiny : yLo + (j + 1) * bw;
        const mass = bucketMass(segs, lo, hi);
        if (mass > maxMass) maxMass = mass;
        return { lo, hi: yLo + (j + 1) * bw, mass };
      });
      return { x: p.x, p50: p.p50, buckets };
    });

    const colW = W / pts.length;
    const halfMax = (colW / 2) * 0.86;
    const hw = (mass) => (mass / (maxMass || 1)) * halfMax;

    const violins = perStep.map((s, i) => {
      const cx = i * colW + colW / 2;
      const bars = s.buckets
        .filter((b) => b.mass > 0)
        .map((b) => {
          const halfW = hw(b.mass);
          const yTop = sy(b.hi);
          const yBot = sy(b.lo);
          return {
            x: (cx - halfW).toFixed(2),
            w: (halfW * 2).toFixed(2),
            y: yTop.toFixed(2),
            h: Math.max(yBot - yTop - BAR_GAP, 0.5).toFixed(2),
            title: `advantage ${fmt(b.lo)} – ${fmt(b.hi)}\n${(b.mass * 100).toFixed(1)}% of samples`,
          };
        });
      // Median tick spans the bucket that contains p50.
      const medBucket = s.buckets.find((b) => s.p50 >= b.lo && s.p50 < b.hi);
      const medHW = hw(medBucket ? medBucket.mass : 0);
      return {
        bars,
        cx: cx.toFixed(2),
        medY: sy(s.p50).toFixed(2),
        medX1: (cx - Math.max(medHW, 3)).toFixed(2),
        medX2: (cx + Math.max(medHW, 3)).toFixed(2),
        x: s.x,
        p50: s.p50,
      };
    });

    return {
      violins,
      yLo,
      yHi,
      zeroY: yLo <= 0 && yHi >= 0 ? sy(0) : null,
      firstX: pts[0].x,
      lastX: pts[pts.length - 1].x,
      latestMedian: perStep[perStep.length - 1].p50,
      // Per-violin labels get crowded past ~12 steps; fall back to endpoints.
      showEachLabel: pts.length <= 12,
    };
  });

  function labelFor(v, i) {
    if (labels && labels[i] != null) return labels[i];
    return `step ${v.x}`;
  }
</script>

{#if model}
  <div class="violin-legend">
    <span class="violin-legend-item"><span class="vsw fill"></span>bucket density</span>
    <span class="violin-legend-item"><span class="vsw median"></span>median</span>
  </div>
  <svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" aria-hidden="true">
    {#if model.zeroY != null}
      <line
        x1="0"
        x2={W}
        y1={model.zeroY}
        y2={model.zeroY}
        stroke="#fff"
        stroke-width="0.75"
        stroke-opacity="0.35"
        stroke-dasharray="4 4"
      />
    {/if}
    {#each model.violins as v (v.x)}
      {#each v.bars as b, bi (bi)}
        <rect class="bucket" x={b.x} y={b.y} width={b.w} height={b.h}>
          <title>{b.title}</title>
        </rect>
      {/each}
      <line
        x1={v.medX1}
        x2={v.medX2}
        y1={v.medY}
        y2={v.medY}
        stroke="#fff"
        stroke-width="1.5"
        stroke-opacity="0.85"
      />
    {/each}
  </svg>
  <div class="fan-meta">
    <span>min {fmt(model.yLo)}</span>
    <span>latest median {fmt(model.latestMedian)}</span>
    <span>max {fmt(model.yHi)}</span>
  </div>
  {#if model.showEachLabel}
    <div class="violin-labels" style:grid-template-columns={`repeat(${model.violins.length}, 1fr)`}>
      {#each model.violins as v, i (v.x)}
        <span title={labelFor(v, i)}>{labelFor(v, i)}</span>
      {/each}
    </div>
  {:else}
    <div class="fan-axis">
      <span>step {model.firstX}</span>
      <span class="fan-axis-label">training step</span>
      <span>step {model.lastX}</span>
    </div>
  {/if}
{:else}
  <div class="empty">Advantage distribution needs ≥1 step of data.</div>
{/if}

<style>
  svg {
    width: 100%;
    height: 210px;
    display: block;
    background: #0a0e14;
    border-radius: 4px;
  }
  .bucket {
    fill: var(--accent);
    fill-opacity: 0.35;
    transition: fill-opacity 0.08s ease;
  }
  .bucket:hover {
    fill-opacity: 0.85;
  }
  .violin-legend {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
    font-size: 11px;
    color: var(--muted);
  }
  .violin-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }
  .vsw {
    width: 12px;
    height: 10px;
    border-radius: 2px;
  }
  .vsw.fill {
    background: color-mix(in srgb, var(--accent) 40%, transparent);
    border: 1px solid var(--accent);
  }
  .vsw.median {
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
  .violin-labels {
    display: grid;
    margin-top: 2px;
    font-size: 10px;
    color: var(--muted);
  }
  .violin-labels span {
    text-align: center;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    padding: 0 2px;
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
