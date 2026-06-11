<script>
  // Underlined tab bar, ported from Modal's UnderlinedTabList / UnderlinedTab:
  // a bottom-bordered nav where the active tab gets a green underline + bright
  // text. `tabs` is [{ value, label, count? }]; `active` is bindable.
  let { tabs = [], active = $bindable() } = $props();
</script>

<div class="tab-list" role="tablist">
  {#each tabs as tab (tab.value)}
    <button
      class="tab"
      class:active={active === tab.value}
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

<style>
  .tab-list {
    display: flex;
    border-bottom: 1px solid var(--color-c-gray-10, #2f2f2f);
  }

  .tab {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    margin-bottom: -1px;
    border: 0;
    border-bottom: 2px solid transparent;
    background: none;
    color: var(--muted);
    font: inherit;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition:
      color 0.12s ease,
      border-color 0.12s ease;
  }

  .tab:hover {
    color: var(--text-bright);
  }

  .tab.active {
    color: var(--text-bright);
    border-bottom-color: var(--green, var(--accent));
  }

  .tab-count {
    font-size: 11px;
    line-height: 1;
    color: var(--muted);
    background: var(--color-c-gray-10, #2f2f2f);
    border-radius: 999px;
    padding: 3px 7px;
    font-variant-numeric: tabular-nums;
  }

  .tab.active .tab-count {
    color: var(--text-bright);
  }
</style>
