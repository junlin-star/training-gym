<script>
  let { inference = null } = $props();

  function formatCount(value) {
    return Number.isFinite(value) ? Math.round(value).toLocaleString() : "—";
  }

  function formatPercent(value) {
    return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
  }

  
</script>

{#if inference}
  <div
    class="grid grid-cols-[repeat(auto-fit,minmax(110px,1fr))] gap-[6px] mb-[10px]"
    aria-label="Inference statistics"
  >
    <div class="rounded-[4px] [border:1px_solid_var(--border)] p-[6px_8px]">
      <div class="text-[10px] uppercase tracking-[0.05em] text-(--muted)">tokens in</div>
      <div class="text-[12px] text-(--text-bright) [font-variant-numeric:tabular-nums]">
        {formatCount(inference.tokens_in)}
      </div>
    </div>
    <div class="rounded-[4px] [border:1px_solid_var(--border)] p-[6px_8px]">
      <div class="text-[10px] uppercase tracking-[0.05em] text-(--muted)">tokens out</div>
      <div class="text-[12px] text-(--text-bright) [font-variant-numeric:tabular-nums]">
        {formatCount(inference.tokens_out)}
      </div>
    </div>
    {#if Number.isFinite(inference.new_tokens)}
      <div class="rounded-[4px] [border:1px_solid_var(--border)] p-[6px_8px]">
        <div class="text-[10px] uppercase tracking-[0.05em] text-(--muted)">new tokens</div>
        <div class="text-[12px] text-(--text-bright) [font-variant-numeric:tabular-nums]">
          {formatCount(inference.new_tokens)}
        </div>
      </div>
    {/if}
    {#if Number.isFinite(inference.cached_tokens)}
      <div class="rounded-[4px] [border:1px_solid_var(--border)] p-[6px_8px]">
        <div class="text-[10px] uppercase tracking-[0.05em] text-(--muted)">cached tokens</div>
        <div class="text-[12px] text-(--text-bright) [font-variant-numeric:tabular-nums]">
          {formatCount(inference.cached_tokens)}
          <span class="text-(--muted)">({formatPercent(inference.cache_hit_rate)})</span>
        </div>
      </div>
    {/if}
  
  </div>
{/if}
