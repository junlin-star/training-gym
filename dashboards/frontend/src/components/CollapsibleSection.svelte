<script>
  import { ChevronUp } from "lucide-svelte";
  import { slide } from "svelte/transition";

  // Port of Modal's CollapsibleDrawerSection: a top-bordered panel with a
  // clickable header (chevron rotates when open) and a slide-animated body.
  // `title` and `body` are snippets so callers control the header content.
  let { isOpen = $bindable(true), title, body } = $props();
</script>

<div class="collapsible">
  <button
    class="collapsible-header"
    onclick={() => (isOpen = !isOpen)}
    aria-expanded={isOpen}
  >
    <div class="collapsible-title">{@render title()}</div>
    <div class="collapsible-chevron" class:open={isOpen}>
      <ChevronUp size={18} />
    </div>
  </button>

  {#if isOpen}
    <div class="collapsible-body" transition:slide={{ duration: 200 }}>
      {@render body()}
    </div>
  {/if}
</div>

<style>
  .collapsible {
    border-top: 1px solid var(--color-c-gray-10, #2f2f2f);
  }

  .collapsible-header {
    display: flex;
    width: 100%;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 12px 0;
    border: 0;
    background: none;
    color: inherit;
    font: inherit;
    cursor: pointer;
    text-align: left;
  }

  .collapsible-title {
    flex: 1;
    min-width: 0;
  }

  .collapsible-chevron {
    flex-shrink: 0;
    color: var(--muted);
    transition: transform 0.2s ease;
  }

  .collapsible-chevron.open {
    transform: rotate(180deg);
  }

  .collapsible-body {
    padding-bottom: 16px;
  }
</style>
