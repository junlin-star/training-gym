<script>
  // Per-sample execution timeline: a small Gantt of the spans slime recorded
  // while producing one rollout sample (generate / reward / tool calls / agent
  // steps). `trace` is a list of {name, start, end, attributes, parent} with
  // times in seconds, already rebased so the first span starts at 0. A span
  // with `end == null` is an instant event and renders as a point marker.
  let { trace = [] } = $props();

  let spans = $derived(Array.isArray(trace) ? trace : []);

  let domainMax = $derived.by(() => {
    let max = 0;
    for (const s of spans) {
      const start = Number(s.start) || 0;
      const end = s.end == null ? start : Number(s.end) || start;
      if (end > max) max = end;
    }
    return max || 1;
  });

  const COLORS = [
    "var(--accent)",
    "#60a5fa",
    "#f59e0b",
    "#4ade80",
    "#f472b6",
    "#a78bfa",
  ];

  // Stable color per span name so the same span type reads the same hue across
  // every row (and across samples).
  let colorFor = $derived.by(() => {
    const names = [...new Set(spans.map((s) => s.name || ""))];
    const map = new Map();
    names.forEach((n, i) => map.set(n, COLORS[i % COLORS.length]));
    return (name) => map.get(name || "") || COLORS[0];
  });

  function durLabel(s) {
    const start = Number(s.start) || 0;
    if (s.end == null) return `@${start.toFixed(3)}s`;
    const d = (Number(s.end) || start) - start;
    return `${d.toFixed(3)}s`;
  }

  function rowTitle(s) {
    const a = s.attributes || {};
    const keys = Object.keys(a);
    const attrs = keys.length
      ? " · " + keys.map((k) => `${k}=${a[k]}`).join(", ")
      : "";
    const parent = s.parent ? ` · in ${s.parent}` : "";
    return `${s.name || "span"} · ${durLabel(s)}${parent}${attrs}`;
  }
</script>

{#if spans.length}
  <div class="timeline">
    {#each spans as s, i (i)}
      {@const start = Number(s.start) || 0}
      {@const end = s.end == null ? start : Number(s.end) || start}
      {@const left = (start / domainMax) * 100}
      {@const width =
        s.end == null ? 0 : Math.max(((end - start) / domainMax) * 100, 0.6)}
      <div class="tl-row" title={rowTitle(s)}>
        <span class="tl-label" class:child={!!s.parent}>{s.name || "—"}</span>
        <div class="tl-track">
          {#if s.end == null}
            <span
              class="tl-point"
              style:left={`${left}%`}
              style:background={colorFor(s.name)}
            ></span>
          {:else}
            <span
              class="tl-bar"
              style:left={`${left}%`}
              style:width={`${width}%`}
              style:background={colorFor(s.name)}
            ></span>
          {/if}
        </div>
        <span class="tl-dur">{durLabel(s)}</span>
      </div>
    {/each}
    <div class="tl-axis">
      <span>0s</span>
      <span>{domainMax.toFixed(3)}s</span>
    </div>
  </div>
{:else}
  <div class="tl-empty">No trace recorded for this sample.</div>
{/if}

<style>
  .timeline {
    background: var(--color-c-gray-08, #1c1c1c);
    border-radius: 4px;
    padding: 8px 10px;
    max-height: 260px;
    overflow-y: auto;
  }

  .tl-row {
    display: grid;
    grid-template-columns: 120px minmax(0, 1fr) 64px;
    align-items: center;
    gap: 8px;
    height: 18px;
  }

  .tl-label {
    font-size: 11px;
    color: var(--text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .tl-label.child {
    padding-left: 8px;
    color: var(--muted);
  }

  .tl-track {
    position: relative;
    height: 10px;
    border-radius: 2px;
    background: var(--color-c-gray-10, #2f2f2f);
  }

  .tl-bar {
    position: absolute;
    top: 0;
    height: 10px;
    min-width: 2px;
    border-radius: 2px;
  }

  .tl-point {
    position: absolute;
    top: 1px;
    width: 8px;
    height: 8px;
    margin-left: -4px;
    border-radius: 9999px;
  }

  .tl-dur {
    font-size: 10px;
    color: var(--muted);
    text-align: right;
    font-variant-numeric: tabular-nums;
  }

  .tl-axis {
    display: flex;
    justify-content: space-between;
    margin-top: 6px;
    padding-top: 4px;
    border-top: 1px solid var(--border, #2f2f2f);
    font-size: 10px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .tl-empty {
    font-size: 12px;
    color: var(--muted);
    padding: 4px 0;
  }
</style>
