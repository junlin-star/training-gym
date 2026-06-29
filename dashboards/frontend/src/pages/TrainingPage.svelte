<script>
  import { ExternalLink, Maximize2, PanelRightClose } from "lucide-svelte";
  import Drawer from "../components/Drawer.svelte";
  import FilterBar from "../components/FilterBar.svelte";
  import FrameworkStageProgress from "../components/FrameworkStageProgress.svelte";
  import MinimalTableSkeleton from "../components/MinimalTableSkeleton.svelte";
  import ResizableTable from "../components/ResizableTable.svelte";
  import RunSummary from "../components/RunSummary.svelte";
  import StatusPill from "../components/StatusPill.svelte";
  import TimeAgo from "../components/TimeAgo.svelte";
  import { formatTagValue, getGroupTags, smoothedStageLabel } from "../lib/format.js";

  let {
    allRuns,
    completedTotal,
    runningTotal,
    stoppedTotal,
    failedTotal,
    recipes,
    recipeCounts,
    activeRecipes,

    statuses,
    statusCounts,
    activeStatuses,
    groups,
    groupCounts,
    activeGroups,
    filteredRuns,
    loading,
    error,
    modelName,
    getStatus,
    getFrameworkStatus,
    showFrameworkStatus,
    fmtDuration,
    search = $bindable(),
    drawerRunId = null,
    onOpenDetail = () => {},
    onCloseDrawer = () => {},
    onToggleRecipe,
    onToggleAllRecipes,
    onToggleStatus,
    onToggleGroup,
    onToggleAllGroups,
  } = $props();

  // The drawer is now driven by the parent: it holds the run-summary while the
  // full rollouts/logs detail lives on its own page. Clicking a run (or the
  // drawer's Expand button) navigates to that page; the page's Collapse button
  // brings the summary back as this drawer.
  let selectedRun = $derived.by(
    () => allRuns.find((run) => run.run_id === drawerRunId) || null,
  );

  const drawerWidth = "min(420px, calc(100vw - 24px))";
  const columns = [
    { key: "name", label: "Name", width: 240, minWidth: 140 },
    { key: "status", label: "Status", width: 116, minWidth: 96 },
    { key: "stage", label: "Stage", width: 190, minWidth: 130 },
    { key: "model", label: "Model", width: 210, minWidth: 120 },
    { key: "dataset", label: "Dataset", width: 180, minWidth: 120 },
    { key: "recipe", label: "Recipe", width: 116, minWidth: 88 },
    { key: "group", label: "Group", width: 150, minWidth: 96 },
    { key: "tags", label: "Tags", width: 220, minWidth: 140 },
    { key: "created", label: "Created", width: 116, minWidth: 96 },
    { key: "updated", label: "Last updated", width: 132, minWidth: 110 },
    { key: "actions", label: "", ariaLabel: "Actions", width: 236, minWidth: 180 },
  ];

  function selectRun(runId) {
    onOpenDetail(runId);
  }

  function closeDrawer() {
    onCloseDrawer();
  }

  function frameworkProgress(run) {
    const progress = run?.framework_progress;
    if (!progress || typeof progress !== "object") return null;
    const current = Number(progress.current);
    const total = Number(progress.total);
    if (!Number.isFinite(current) || !Number.isFinite(total) || total <= 0) {
      return null;
    }
    return {
      current: Math.max(0, Math.min(current, total)),
      total,
      unit: progress.unit || "step",
    };
  }

  function frameworkStatusLabel(run) {
    // Pass the raw `framework_progress` so smoothedStageLabel sees is_active
    // even for stages that don't have step counters yet (download/convert).
    return smoothedStageLabel(getFrameworkStatus(run), run?.framework_progress);
  }

  function progressLabel(progress) {
    if (!progress) return "";
    const unit = String(progress.unit || "step");
    const label = unit.charAt(0).toUpperCase() + unit.slice(1);
    return `${label} ${progress.current} / ${progress.total}`;
  }

  $effect(() => {
    if (drawerRunId && !allRuns.some((run) => run.run_id === drawerRunId)) {
      onCloseDrawer();
    }
  });
</script>

<section class="summary-row training-summary">
  <article class="summary-card">
    <span class="summary-label">Total runs</span>
    <strong>{allRuns.length}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Completed runs</span>
    <strong>{completedTotal}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Pending runs</span>
    <strong>{runningTotal}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Stopped runs</span>
    <strong>{stoppedTotal}</strong>
  </article>
  <article class="summary-card">
    <span class="summary-label">Failed runs</span>
    <strong>{failedTotal}</strong>
  </article>
</section>

<section class="runs-surface">
  <div class="filters-row">
    <FilterBar
      {recipes}
      {recipeCounts}
      {activeRecipes}
      allRecipesActive={activeRecipes.size === recipes.length}
      {statuses}
      {statusCounts}
      {activeStatuses}
      {groups}
      {groupCounts}
      {activeGroups}
      allGroupsActive={activeGroups.size === groups.length}
      totalRuns={allRuns.length}
      bind:search
      onToggleRecipe={onToggleRecipe}
      onToggleAllRecipes={onToggleAllRecipes}
      onToggleStatus={onToggleStatus}
      onToggleGroup={onToggleGroup}
      onToggleAllGroups={onToggleAllGroups}
    />
  </div>

  <div class="runs-body">
    {#if loading}
      <div class="table-wrap">
        <MinimalTableSkeleton
          class="runs-table"
          columns={["Name", "Status", "Stage", "Model", "Dataset", "Recipe", "Group", "Tags", "Created", "Last updated", ""]}
          rows={8}
        />
      </div>
    {:else if error}
      <div class="empty">Failed to load: {error}</div>
    {:else if !allRuns.length}
      <div class="empty">No training runs found yet.</div>
    {:else if !filteredRuns.length}
      <div class="empty">No runs match the current filters.</div>
    {:else}
      <div class="table-wrap">
        <ResizableTable class="runs-table training-runs-table" {columns} stickyFirstColumn>
          <tbody>
            {#each filteredRuns as run, runIndex (`${run.run_id || "run"}-${run.created_at || 0}-${runIndex}`)}
              {@const runName = run.run_id || "—"}
              {@const status = getStatus(run)}
              {@const stageLabel = frameworkStatusLabel(run)}
              {@const progress = frameworkProgress(run)}
              {@const groupTags = getGroupTags(run)}
              <tr class:row-selected={drawerRunId === run.run_id}>
                <td class="run-cell">
                  <button
                    class="cell-open-button run-name-button"
                    title={runName}
                    onclick={() => selectRun(run.run_id)}
                  >
                    <div class="run-name">{runName}</div>
                  </button>
                </td>
                <td>
                  <button class="cell-open-button" onclick={() => selectRun(run.run_id)}>
                    <StatusPill status={status} />
                  </button>
                </td>
                <td class="stage-cell">
                  <button
                    class="cell-open-button stage-open-button"
                    onclick={() => selectRun(run.run_id)}
                  >
                    {#if showFrameworkStatus(run) && stageLabel}
                      <FrameworkStageProgress
                        progress={progress}
                        progressLabel={progressLabel(progress)}
                        stageLabel={stageLabel}
                        compact
                        active={status.toLowerCase() === "pending"}
                      />
                    {:else}
                      <span class="stage-empty">—</span>
                    {/if}
                  </button>
                </td>
                <td class="model-cell" title={modelName(run)}>
                  <button class="cell-open-button" onclick={() => selectRun(run.run_id)}>
                    {modelName(run)}
                  </button>
                </td>
                <td class="dataset-cell" title={run.config_summary?.dataset_name || "—"}>
                  <button class="cell-open-button" onclick={() => selectRun(run.run_id)}>
                    {run.config_summary?.dataset_name || "—"}
                  </button>
                </td>
                <td>
                  <button class="cell-open-button" onclick={() => selectRun(run.run_id)}>
                    {run.framework || "—"}
                  </button>
                </td>
                <td class="group-cell" title={groupTags?.group_id || run.group_id || ""}>
                  <button class="cell-open-button" onclick={() => selectRun(run.run_id)}>
                    {#if groupTags?.group_id}
                      <span class="group-tag">{groupTags.group_id}</span>
                    {:else}
                      <span class="group-empty">—</span>
                    {/if}
                  </button>
                </td>
                <td class="tags-cell">
                  <button class="cell-open-button tags-open-button" onclick={() => selectRun(run.run_id)}>
                    {#if groupTags?.tags.length}
                      <span class="tag-pill-list">
                        {#each groupTags.tags as tag (tag.key)}
                          <span class="tag-pill" title={`${tag.key}=${formatTagValue(tag.value)}`}>
                            <span class="tag-pill-key">{tag.key}</span><span>=</span><span class="tag-pill-value">{formatTagValue(tag.value)}</span>
                          </span>
                        {/each}
                      </span>
                    {:else}
                      <span class="group-empty">—</span>
                    {/if}
                  </button>
                </td>
                <td class="created-cell">
                  <TimeAgo timestamp={run.started_at || run.created_at} showJustNow falsyRepresentation="—" />
                </td>
                <td class="updated-cell">
                  <TimeAgo timestamp={run.updated_at} showJustNow falsyRepresentation="—" />
                </td>
                <td class="modal-link-cell">
                  <div class="modal-link-wrap">
                    <button
                      class="expand-button"
                      title="Open expanded view"
                      aria-label={`Open expanded view for training run ${run.run_id}`}
                      onclick={(event) => {
                        event.stopPropagation();
                        onOpenDetail(run.run_id);
                      }}
                    >
                      <Maximize2 size={12} strokeWidth={2.1} />
                      <span class="expand-button-label">Expand</span>
                    </button>
                    {#if run.modal_app_url}
                      <a
                        class="open-modal-link"
                        href={run.modal_app_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        onclick={(event) => event.stopPropagation()}
                      >
                        <span class="open-modal-link-label">Open in Modal</span>
                        <ExternalLink class="open-modal-link-icon" size={12} strokeWidth={2.1} />
                      </a>
                    {:else}
                      <span class="open-modal-link open-modal-link-disabled">
                        <span class="open-modal-link-label">Open in Modal</span>
                        <ExternalLink class="open-modal-link-icon" size={12} strokeWidth={2.1} />
                      </span>
                    {/if}
                  </div>
                </td>
              </tr>
            {/each}
          </tbody>
        </ResizableTable>
      </div>
    {/if}
  </div>
</section>

{#if selectedRun}
  <Drawer open={!!selectedRun} onclose={closeDrawer} width={drawerWidth}>
    <div
      class="run-drawer"
      aria-label={`Training run ${selectedRun.run_id}`}
    >
      <div class="drawer-header">
        <div class="drawer-header-left">
          <div class="drawer-eyebrow">Training run</div>
          <h2 class="drawer-run-id" title={selectedRun.run_id}>{selectedRun.run_id}</h2>
        </div>
        <div class="drawer-actions">
          <button
            class="drawer-expand-button"
            onclick={() => onOpenDetail(selectedRun.run_id)}
            title="Expand to full view"
          >
            <Maximize2 size={12} />
            <span>Expand</span>
          </button>
          {#if selectedRun.modal_app_url}
            <a
              class="view-app-link"
              href={selectedRun.modal_app_url}
              target="_blank"
              rel="noopener noreferrer"
            >
              <span>View in app</span>
              <ExternalLink size={12} />
            </a>
          {/if}
          <button class="drawer-close" onclick={closeDrawer} aria-label="Close run drawer">
            <PanelRightClose size={16} />
          </button>
        </div>
      </div>

      <div class="drawer-summary">
        <RunSummary
          run={selectedRun}
          {getStatus}
          {showFrameworkStatus}
          {getFrameworkStatus}
          {modelName}
          {fmtDuration}
        />
      </div>
    </div>
  </Drawer>
{/if}

<style>
  .summary-row {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 14px;
    margin-bottom: 24px;
  }

  .training-summary {
    margin-bottom: 24px;
  }

  .summary-card {
    border: 0;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.03);
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    height: 80px;
  }

  .summary-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 400;
    line-height: 16px;
  }

  .summary-card strong {
    color: var(--text-bright);
    font-size: 20px;
    font-weight: 500;
    line-height: 32px;
  }

  .runs-surface {
    border: 0;
    background: transparent;
    display: flex;
    flex-direction: column;
    gap: 24px;
    padding: 0;
  }

  .filters-row {
    margin: 0;
  }

  .runs-body {
    padding: 0;
  }

  .table-wrap {
    max-width: 100%;
    overflow: auto hidden;
    overscroll-behavior-x: contain;
    overscroll-behavior-y: auto;
    -webkit-overflow-scrolling: auto;
  }

  :global(table.runs-table) {
    width: 100%;
    min-width: 960px;
  }

  :global(table.runs-table tr.row-selected td) {
    background: color-mix(in srgb, var(--accent) 6%, transparent);
  }

  .cell-open-button {
    border: 0;
    background: transparent;
    color: var(--text);
    font: inherit;
    font-size: 14px;
    line-height: 20px;
    cursor: pointer;
    padding: 0;
    text-align: left;
    width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .cell-open-button:hover {
    color: var(--accent);
  }

  .stage-open-button {
    white-space: normal;
    line-height: 16px;
  }

  .run-name {
    display: block;
    color: var(--text-bright);
    font-family: var(--font-mono);
    font-weight: 400;
    font-size: 14px;
    line-height: 20px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .run-cell {
    min-width: 0;
  }

  .model-cell {
    min-width: 0;
  }

  .dataset-cell {
    min-width: 0;
  }

  .stage-cell {
    min-width: 0;
  }

  .created-cell,
  .updated-cell {
    white-space: nowrap;
  }

  .stage-empty {
    color: var(--muted);
  }

  .group-cell {
    max-width: 12rem;
  }

  .tags-cell {
    max-width: 16rem;
  }

  .group-tag {
    display: inline-block;
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    vertical-align: bottom;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-variant-numeric: tabular-nums;
    color: var(--muted);
    border: 1px solid var(--border, #2f2f2f);
    background: color-mix(in srgb, var(--panel-alt) 70%, transparent);
  }

  .tags-open-button {
    white-space: normal;
  }

  .tag-pill-list {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    min-width: 0;
  }

  .tag-pill {
    display: inline-flex;
    align-items: baseline;
    max-width: 100%;
    min-width: 0;
    overflow: hidden;
    padding: 2px 7px;
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 999px;
    background: color-mix(in srgb, var(--panel-alt) 70%, transparent);
    color: var(--text);
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 14px;
  }

  .tag-pill-key,
  .tag-pill-value {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .tag-pill-key {
    color: var(--muted);
  }

  .group-empty {
    color: var(--muted);
  }

  .modal-link-cell {
    min-width: 0;
    overflow: visible;
  }

  .modal-link-wrap {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .expand-button {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
    border: 1px solid var(--color-c-gray-10, #2f2f2f);
    border-radius: 6px;
    padding: 4px 8px;
    font: inherit;
    font-size: 12px;
    font-weight: 500;
    line-height: 16px;
    color: var(--muted);
    background: transparent;
    cursor: pointer;
  }

  .expand-button:hover {
    color: var(--text-bright);
    border-color: var(--border-strong);
  }

  .open-modal-link {
    display: inline-flex;
    align-items: center;
    justify-content: flex-start;
    gap: 6px;
    white-space: nowrap;
    flex-shrink: 0;
    border: 1px solid var(--color-c-gray-10, #2f2f2f);
    border-radius: 6px;
    padding: 4px 8px;
    text-decoration: none;
    font-size: 12px;
    font-weight: 500;
    line-height: 16px;
    background: transparent;
  }

  .open-modal-link-label {
    color: var(--muted);
    overflow: hidden;
    text-overflow: ellipsis;
  }

  :global(.open-modal-link-icon) {
    color: var(--muted-strong);
  }

  .open-modal-link:hover {
    border-color: var(--border-strong);
  }

  .open-modal-link-disabled {
    opacity: 0.5;
    pointer-events: none;
  }

  .run-drawer {
    width: 100%;
    height: 100%;
    max-height: 100vh;
  }

  .drawer-summary {
    padding: 4px 20px 16px;
  }

  .drawer-header {
    border-bottom: 1px solid var(--color-c-gray-10, #2f2f2f);
    padding: 16px 20px;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
  }

  .drawer-header-left {
    min-width: 0;
    overflow: hidden;
  }

  .drawer-eyebrow {
    color: var(--muted);
    font-size: 12px;
    line-height: 16px;
    margin-bottom: 4px;
  }

  .drawer-run-id {
    color: var(--text-bright);
    font-size: 16px;
    font-weight: 500;
    font-family: var(--font-mono);
    line-height: 24px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .drawer-actions {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
  }

  .view-app-link {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border: 1px solid var(--color-c-gray-10, #2f2f2f);
    border-radius: 6px;
    padding: 4px 8px;
    text-decoration: none;
    color: var(--muted);
    font-size: 12px;
    font-weight: 500;
    line-height: 16px;
  }

  .view-app-link:hover {
    color: var(--text-bright);
    border-color: var(--border-strong);
  }

  .drawer-expand-button {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border: 1px solid var(--color-c-gray-10, #2f2f2f);
    border-radius: 6px;
    padding: 4px 8px;
    font: inherit;
    font-size: 12px;
    font-weight: 500;
    line-height: 16px;
    color: var(--muted);
    background: transparent;
    cursor: pointer;
  }

  .drawer-expand-button:hover {
    color: var(--text-bright);
    border-color: var(--border-strong);
  }

  .drawer-close {
    border: 1px solid var(--color-c-gray-10, #2f2f2f);
    border-radius: 6px;
    background: transparent;
    color: var(--muted);
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 4px;
  }

  .drawer-close:hover {
    color: var(--text-bright);
    border-color: var(--border-strong);
  }

  .empty {
    padding: 24px;
    color: var(--muted);
    text-align: center;
    font-size: 0.84rem;
  }

  @media (max-width: 900px) {
    .summary-row {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }

  @media (max-width: 640px) {
    .summary-row {
      grid-template-columns: 1fr;
    }
  }
</style>
