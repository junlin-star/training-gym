<script>
  // Underlined tab bar, ported from Modal's UnderlinedTabList / UnderlinedTab:
  // a bottom-bordered nav where the active tab gets a green underline + bright
  // text. `tabs` is [{ value, label, count? }]; `active` is bindable.
  let { tabs = [], active = $bindable() } = $props();
</script>

<div class="flex [border-bottom:1px_solid_var(--color-c-gray-10,#2f2f2f)]" role="tablist">
  {#each tabs as tab (tab.value)}
    <button
      class="tab"
      class:tabs-active={active === tab.value}
      role="tab"
      aria-selected={active === tab.value}
      onclick={() => (active = tab.value)}
    >
      <span>{tab.label}</span>
      {#if tab.count != null}
        <span class="tab-count">{tab.count}</span>
      {/if}
    </button>
  {/each}
</div>
