<script>
  import { onDestroy } from "svelte";
  import MinimalTable from "./MinimalTable.svelte";

  let {
    columns = [],
    stickyFirstColumn = false,
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

  function stopFrozenColumnHorizontalScroll(event) {
    if (!stickyFirstColumn || Math.abs(event.deltaX) < 4 || Math.abs(event.deltaY) > 1) return;
    const firstColumnCell = event.target?.closest?.("th:first-child, td:first-child");
    if (!firstColumnCell) return;
    event.preventDefault();
    event.stopPropagation();
  }

  onDestroy(() => {
    stopColumnResize();
  });
</script>

<MinimalTable
  {...restProps}
  class={`resizable-table ${stickyFirstColumn ? "sticky-first-column" : ""} ${classOverride}`.trim()}
  onwheel={stopFrozenColumnHorizontalScroll}
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
    right: -9px;
    bottom: 0;
    z-index: 1;
    width: 18px;
    border: 0;
    padding: 0;
    background: transparent;
    cursor: col-resize;
  }

  .column-resize-handle::after {
    content: "↔";
    position: absolute;
    top: 50%;
    right: 1px;
    display: grid;
    place-items: center;
    width: 16px;
    height: 16px;
    border-radius: 999px;
    background: var(--color-c-gray-10, #2f2f2f);
    color: var(--muted);
    font-size: 11px;
    line-height: 1;
    opacity: 0;
    transform: translateY(-50%);
    transition:
      opacity 120ms ease,
      color 120ms ease,
      background 120ms ease;
  }

  :global(table.resizable-table th:hover) .column-resize-handle::after,
  :global(table.resizable-table th.resizing) .column-resize-handle::after,
  .column-resize-handle:focus-visible::after {
    opacity: 1;
  }

  :global(table.resizable-table th.resizing) .column-resize-handle::after,
  .column-resize-handle:focus-visible::after {
    background: var(--color-c-gray-15, #3b3b3b);
    color: var(--text-bright);
  }

  .column-resize-handle:focus-visible {
    outline: none;
  }

  :global(table.resizable-table.sticky-first-column th:first-child),
  :global(table.resizable-table.sticky-first-column tbody td:first-child) {
    position: sticky;
    left: 0;
    background: var(--bg);
    background-clip: padding-box;
    border-right: 1px solid var(--color-c-gray-10, #2f2f2f);
  }

  :global(table.resizable-table.sticky-first-column th:first-child) {
    z-index: 3;
  }

  :global(table.resizable-table.sticky-first-column tbody td:first-child) {
    z-index: 2;
  }

  :global(table.resizable-table.sticky-first-column tbody tr:hover td:first-child) {
    background: color-mix(in srgb, var(--text) 6%, var(--bg));
  }

  :global(table.resizable-table.sticky-first-column tr.row-selected td:first-child) {
    background: color-mix(in srgb, var(--text) 8%, var(--bg));
  }
</style>
