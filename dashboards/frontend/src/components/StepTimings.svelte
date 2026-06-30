<script>
  // Per-step + per-substep timing breakdown. `stepTimes` and `substepTimes`
  // are populated at the end of a run (rank 0 flushes them), so this renders
  // nothing until the data lands.
  let { stepTimes = null, substepTimes = null, fmtDuration } = $props();

  const SUBSTEP_LABELS = {
    evaluate_rollouts: "Eval (before)",
    generate_rollouts: "Generate rollouts",
    offload_rollout: "Offload rollout",
    compute_log_probs: "Compute log probs",
    optimizer_step: "Optimizer step",
    weight_sync: "Weight sync",
    checkpoint_save: "Checkpoint save",
    offload_train: "Offload train",
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

  const ORDER = Object.keys(SUBSTEP_LABELS);

  function labelFor(name) {
    return SUBSTEP_LABELS[name] || name.replace(/_/g, " ");
  }

  function colorFor(name) {
    return SUBSTEP_COLORS[name] || "var(--color-c-gray-40, #5e5e5e)";
  }

  function fmtSecs(s) {
    if (s == null) return "—";
    if (typeof fmtDuration === "function") return fmtDuration(0, s);
    return `${s}s`;
  }

  let steps = $derived.by(() => {
    const stepKeys = Object.keys(stepTimes || {});
    const subKeys = Object.keys(substepTimes || {});
    const keys = Array.from(new Set([...stepKeys, ...subKeys]));
    const out = keys.map((k) => {
      const st = (stepTimes || {})[k] || null;
      const subs = (substepTimes || {})[k] || {};
      const substeps = Object.entries(subs)
        .map(([name, v]) => ({
          name,
          start: v?.start ?? null,
          duration: v?.duration_s ?? null,
        }))
        .sort((a, b) => (a.start ?? 0) - (b.start ?? 0));
      return {
        key: k,
        n: Number.isFinite(Number(k)) ? Number(k) : k,
        duration: st?.duration_s ?? null,
        substeps,
      };
    });
    out.sort((a, b) => (Number(a.key) || 0) - (Number(b.key) || 0));
    return out;
  });

  let hasData = $derived(steps.length > 0);

  let legend = $derived.by(() => {
    const seen = new Set();
    for (const s of steps) for (const sub of s.substeps) seen.add(sub.name);
    return ORDER.filter((n) => seen.has(n));
  });
</script>

{#if hasData}
  <div class="step-timings">
    {#if legend.length}
      <div class="legend">
        {#each legend as name (name)}
          <span class="legend-item">
            <span class="swatch" style:background={colorFor(name)}></span>
            {labelFor(name)}
          </span>
        {/each}
      </div>
    {/if}

    {#each steps as step (step.key)}
      <div class="step-row">
        <div class="step-head">
          <span class="step-name">Step {step.n}</span>
          <span class="step-dur">{fmtSecs(step.duration)}</span>
        </div>
        {#if step.substeps.length}
          <div class="bar">
            {#each step.substeps as sub (sub.name)}
              {#if sub.duration == null}
                <div
                  class="seg seg-null"
                  title={`${labelFor(sub.name)}: unknown (report dropped)`}
                ></div>
              {:else}
                <div
                  class="seg"
                  style:flex-grow={Math.max(sub.duration, 0.01)}
                  style:background={colorFor(sub.name)}
                  title={`${labelFor(sub.name)}: ${fmtSecs(sub.duration)}`}
                ></div>
              {/if}
            {/each}
          </div>
        {:else}
          <div class="bar bar-empty"></div>
        {/if}
      </div>
    {/each}
  </div>
{/if}

<style>
  .step-timings {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 14px;
    margin-bottom: 4px;
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

  .step-row {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .step-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
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
</style>
