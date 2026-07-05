<script>
  import { RefreshCw } from "lucide-svelte";

  let { title, statusText, refreshing = false, onRefresh } = $props();

  // Spin while a fetch is in flight, with a short tail after it finishes so a
  // fast refresh still completes a smooth rotation instead of jerking to a stop.
  let spinning = $state(false);
  $effect(() => {
    if (refreshing) {
      spinning = true;
      return;
    }
    if (!spinning) return;
    const t = setTimeout(() => (spinning = false), 500);
    return () => clearTimeout(t);
  });
</script>

<header class="flex justify-between items-center mb-[24px]">
  <h1 class="text-(--text) text-[24px] font-medium leading-[36px]">{title}</h1>
  <div class="flex items-center gap-[0.7rem]">
    {#if statusText}
      <span class="text-(--muted) text-[0.76rem] lowercase">{statusText}</span>
    {/if}
    <button class="[border:1px_solid_var(--border-strong)] rounded-[6px] text-(--text) bg-(--bg) [font:inherit] text-[14px] font-medium p-[6px_8px] cursor-pointer inline-flex items-center gap-[8px] hover:[border-color:var(--color-c-gray-50,#8b8b8b)] hover:text-(--text-bright)" onclick={onRefresh}>
      <span class="refresh-icon" class:spinning>
        <RefreshCw size={16} strokeWidth={2.1} />
      </span>
      <span>Refresh</span>
    </button>
  </div>
</header>
