<script>
  import { onDestroy } from "svelte";
  import MinimalTable from "./MinimalTable.svelte";

  let {
    columns = [],
    class: classOverride = "",
    style: styleOverride = "",
    ...restProps
  } = $props();

  let columnWidths = $state({});
  let resizeState = $state(null);
  let tableWidth = $derived(
    columns.reduce((total, column) => total + columnWidth(column), 0),
  );

  function columnWidth(column) {
    return columnWidths[column.key] ?? column.width;
  }

  function startColumnResize(event, column) {
    event.preventDefault();
    event.stopPropagation();

    resizeState = {
      key: column.key,
      minWidth: column.minWidth,
      startX: event.clientX,
      startWidth: columnWidth(column),
      previousUserSelect: document.body.style.userSelect,
      previousCursor: document.body.style.cursor,
    };

    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";
    window.addEventListener("pointermove", resizeColumn);
    window.addEventListener("pointerup", stopColumnResize, { once: true });
    window.addEventListener("pointercancel", stopColumnResize, { once: true });
  }

  function resizeColumn(event) {
    if (!resizeState) return;
    const nextWidth = Math.max(
      resizeState.minWidth,
      Math.round(resizeState.startWidth + event.clientX - resizeState.startX),
    );
    columnWidths = { ...columnWidths, [resizeState.key]: nextWidth };
  }

  function stopColumnResize() {
    if (!resizeState) return;
    document.body.style.userSelect = resizeState.previousUserSelect;
    document.body.style.cursor = resizeState.previousCursor;
    resizeState = null;
    window.removeEventListener("pointermove", resizeColumn);
    window.removeEventListener("pointerup", stopColumnResize);
    window.removeEventListener("pointercancel", stopColumnResize);
  }

  onDestroy(() => {
    stopColumnResize();
  });
</script>

<MinimalTable
  {...restProps}
  class={`resizable-table ${classOverride}`.trim()}
  style={`--resizable-grid-width: ${tableWidth}px; width: max(100%, var(--resizable-grid-width)); min-width: var(--resizable-grid-width); ${styleOverride}`.trim()}
>
  <colgroup>
    {#each columns as column (column.key)}
      <col style={`width: ${columnWidth(column)}px;`} />
    {/each}
  </colgroup>
  <thead>
    <tr>
      {#each columns as column (column.key)}
        <th class:resizing={resizeState?.key === column.key}>
          <span class="column-label">{column.label}</span>
          <button
            type="button"
            class="column-resize-handle"
            aria-label={`Resize ${column.ariaLabel || column.label || "column"} column`}
            onpointerdown={(event) => startColumnResize(event, column)}
          ></button>
        </th>
      {/each}
    </tr>
  </thead>
  <!-- svelte-ignore slot_element_deprecated -->
  <slot />
</MinimalTable>

<style>
  :global(table.resizable-table) {
    table-layout: fixed;
    width: max(100%, var(--resizable-grid-width, 960px));
    min-width: var(--resizable-grid-width, 960px);
  }

  :global(table.resizable-table th) {
    position: relative;
    padding-right: 22px;
    user-select: none;
  }

  .column-label {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .column-resize-handle {
    position: absolute;
    top: 0;
    right: -5px;
    bottom: 0;
    z-index: 1;
    width: 10px;
    border: 0;
    padding: 0;
    background: transparent;
    cursor: col-resize;
  }

  .column-resize-handle::after {
    content: "";
    position: absolute;
    top: 8px;
    right: 4px;
    bottom: 8px;
    width: 1px;
    border-radius: 999px;
    background: transparent;
  }

  :global(table.resizable-table th:hover) .column-resize-handle::after,
  :global(table.resizable-table th.resizing) .column-resize-handle::after,
  .column-resize-handle:focus-visible::after {
    background: var(--accent-border);
  }

  .column-resize-handle:focus-visible {
    outline: none;
  }
</style>
