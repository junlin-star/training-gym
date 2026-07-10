<script>
  import { Check, ChevronDown, Filter, Search } from "lucide-svelte";
  import FilterBulkActions from "./FilterBulkActions.svelte";

  let {
    recipes,
    recipeCounts,
    activeRecipes,
    allRecipesActive,
    statuses,
    statusCounts,
    activeStatuses,
    allStatusesActive,
    groups,
    groupCounts,
    activeGroups,
    allGroupsActive,
    search = $bindable(),
    groupBy = $bindable(),
    onToggleRecipe,
    onSelectAllRecipes,
    onClearRecipes,
    onToggleStatus,
    onSelectAllStatuses,
    onClearStatuses,
    onToggleGroup,
    onSelectAllGroups,
    onClearGroups,
  } = $props();

  let openMenu = $state(null);

  function toggleMenu(menu) {
    openMenu = openMenu === menu ? null : menu;
  }
</script>

<svelte:window onclick={() => (openMenu = null)} />

<nav class="p-0 flex items-center gap-[0.5rem] relative flex-wrap">
  <label class="inline-flex items-center gap-[8px] [border:1px_solid_var(--color-c-gray-10,#2f2f2f)] rounded-[6px] [background:transparent] w-[260px] p-[6px_8px]" aria-label="Search training runs by name">
    <span class="search-icon">
      <Search size={13} />
    </span>
    <input
      type="search"
      class="search-input"
      placeholder="Search"
      bind:value={search}
      autocomplete="off"
      spellcheck="false"
    />
  </label>

  <div class="filterbar-menu-wrap">
    <button
      class="filter-button ghost-hover"
      class:filterbar-open={openMenu === "status"}
      onclick={(event) => {
        event.stopPropagation();
        toggleMenu("status");
      }}
    >
      <span class="button-icon">
        <Filter size={12} />
      </span>
      <span>Status</span>
      <span class="chevron" class:rotated={openMenu === "status"}>
        <ChevronDown size={12} />
      </span>
    </button>
    {#if openMenu === "status"}
      <div class="menu">
        <FilterBulkActions
          allSelected={allStatusesActive}
          noneSelected={activeStatuses.size === 0}
          onSelectAll={onSelectAllStatuses}
          onDeselectAll={onClearStatuses}
        />
        {#each statuses as st (st)}
          <button
            class="menu-item"
            onclick={(event) => {
              event.stopPropagation();
              onToggleStatus(st);
            }}
          >
            <span class="checkmark" class:checked={activeStatuses.has(st)}>
              {#if activeStatuses.has(st)}
                <Check size={11} />
              {/if}
            </span>
            <span class="item-label">{st}</span>
            <span class="item-count">{statusCounts[st] || 0}</span>
          </button>
        {/each}
      </div>
    {/if}
  </div>

  <div class="filterbar-menu-wrap">
    <button
      class="filter-button ghost-hover"
      class:filterbar-open={openMenu === "recipes"}
      onclick={(event) => {
        event.stopPropagation();
        toggleMenu("recipes");
      }}
    >
      <span class="button-icon">
        <Filter size={12} />
      </span>
      <span>Recipe</span>
      <span class="chevron" class:rotated={openMenu === "recipes"}>
        <ChevronDown size={12} />
      </span>
    </button>
    {#if openMenu === "recipes"}
      <div class="menu">
        <FilterBulkActions
          allSelected={allRecipesActive}
          noneSelected={activeRecipes.size === 0}
          onSelectAll={onSelectAllRecipes}
          onDeselectAll={onClearRecipes}
        />
        {#each recipes as recipe (recipe)}
          <button
            class="menu-item"
            onclick={(event) => {
              event.stopPropagation();
              onToggleRecipe(recipe);
            }}
          >
            <span class="checkmark" class:checked={activeRecipes.has(recipe)}>
              {#if activeRecipes.has(recipe)}
                <Check size={11} />
              {/if}
            </span>
            <span class="item-label">{recipe}</span>
            <span class="item-count">{recipeCounts[recipe] || 0}</span>
          </button>
        {/each}
      </div>
    {/if}
  </div>

  <div class="filterbar-menu-wrap">
    <button
      class="filter-button ghost-hover"
      class:filterbar-open={openMenu === "groups"}
      onclick={(event) => {
        event.stopPropagation();
        toggleMenu("groups");
      }}
    >
      <span class="button-icon">
        <Filter size={12} />
      </span>
      <span>Group</span>
      <span class="chevron" class:rotated={openMenu === "groups"}>
        <ChevronDown size={12} />
      </span>
    </button>
    {#if openMenu === "groups"}
      <div class="menu">
        <FilterBulkActions
          allSelected={allGroupsActive}
          noneSelected={activeGroups.size === 0}
          onSelectAll={onSelectAllGroups}
          onDeselectAll={onClearGroups}
        />
        {#each groups as group (group)}
          <button
            class="menu-item"
            onclick={(event) => {
              event.stopPropagation();
              onToggleGroup(group);
            }}
          >
            <span class="checkmark" class:checked={activeGroups.has(group)}>
              {#if activeGroups.has(group)}
                <Check size={11} />
              {/if}
            </span>
            <span class="item-label">{group}</span>
            <span class="item-count">{groupCounts[group] || 0}</span>
          </button>
        {/each}
      </div>
    {/if}
  </div>

  <label class="ml-auto inline-flex items-center gap-[6px] text-(--muted) text-[12px] whitespace-nowrap">
    Group by:
    <select class="filter-button ghost-hover" bind:value={groupBy}>
      <option value="none">None</option>
      <option value="group">Group</option>
      <option value="dataset">Dataset</option>
      <option value="model">Model</option>
    </select>
  </label>
</nav>
