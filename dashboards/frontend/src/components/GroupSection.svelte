<script>
  import { ChevronDown } from "lucide-svelte";

  let { title, subtitle, expanded, onToggle, meta, children } = $props();
</script>

<section class="[border:1px_solid_var(--border)] rounded-[8px] [background:transparent] overflow-hidden">
  <button class="[border:0] w-full text-left [background:transparent] text-inherit cursor-pointer flex justify-between items-center gap-[0.8rem] p-[0.75rem_0.9rem]" onclick={onToggle}>
    <div class="min-w-0">
      <div class="text-(--text-bright) text-[0.86rem] [font-weight:600] overflow-hidden text-ellipsis whitespace-nowrap">{title}</div>
      <div class="text-(--muted) text-[0.74rem] overflow-hidden text-ellipsis whitespace-nowrap">{subtitle}</div>
    </div>
    <div class="flex items-center gap-[0.35rem] [flex-wrap:wrap] [justify-content:flex-end]">
      {@render meta?.()}
      <ChevronDown
        size={14}
        style={`color: var(--muted); transform: ${expanded ? "rotate(180deg)" : "rotate(0deg)"};`}
      />
    </div>
  </button>

  {#if expanded}
    {@render children?.()}
  {/if}
</section>
