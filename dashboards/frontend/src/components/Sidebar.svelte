<script>
  let { navItems, activePage, onNavigate } = $props();
</script>

<aside class="[border-right:1px_solid_rgba(255,255,255,0.1)] bg-(--bg-depth) p-[4px_20px_0] max-[900px]:[border-right:0] max-[900px]:[border-bottom:1px_solid_var(--border)]">
  <nav class="flex flex-col gap-[4px] pt-[12px]">
    {#each navItems as item (item.key)}
      <a
        href={item.path}
        class="nav-item"
        class:sidebar-active={activePage === item.key}
        onclick={(event) => {
          event.preventDefault();
          onNavigate(item.key);
        }}
      >
        <span class="inline-flex items-center flex-[0_0_auto] opacity-[0.6]">
          <item.Icon size={14} strokeWidth={2.1} />
        </span>
        <span>{item.label}</span>
      </a>
    {/each}
  </nav>
</aside>
