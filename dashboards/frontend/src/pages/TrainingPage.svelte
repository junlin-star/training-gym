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
    { key: "group", label: "Group", width: 280, minWidth: 220 },
    { key: "tags", label: "Tags", width: 520, minWidth: 360 },
    { key: "created", label: "Created", width: 150, minWidth: 130 },
    { key: "updated", label: "Last updated", width: 170, minWidth: 150 },
    { key: "actions", label: "", ariaLabel: "Actions", width: 236, minWidth: 180 },
  ];

  function trainingRunDetailPath(runId) {
    return `/training/${encodeURIComponent(runId)}`;
  }

  function openRunInNewTab(runId) {
    if (typeof window === "undefined") return;
    const url = new URL(trainingRunDetailPath(runId), window.location.href);
    window.open(url.href, "_blank", "noopener,noreferrer");
  }

  function isPlainLeftClick(event) {
    return event.button === 0 && !event.metaKey && !event.ctrlKey && !event.altKey && !event.shiftKey;
  }

  function selectRun(runId, event) {
    if (event?.shiftKey) {
      event.preventDefault();
      openRunInNewTab(runId);
      return;
    }

    if (event && !isPlainLeftClick(event)) return;
    event?.preventDefault();
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

  function resumeBadge(run) {
    const state = run?.resume_state;
    if (!state) return "";
    const parts = [];
    if (state.attempt_count > 1) parts.push(`attempt ${state.attempt_count}`);
    if (state.resumed_from_checkpoint) {
      parts.push(
        state.resume_from_iteration != null
          ? `resumed @ ${state.resume_from_iteration}`
          : "resumed",
      );
    }
    return parts.join(" · ");
  }

  $effect(() => {
    if (drawerRunId && !allRuns.some((run) => run.run_id === drawerRunId)) {
      onCloseDrawer();
    }
  });
</script>

<section class="summary-sticky grid grid-cols-5 gap-[14px] mb-[24px] max-[900px]:grid-cols-2 max-[640px]:grid-cols-1">
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

<section class="[border:0] [background:transparent] flex flex-col gap-[24px] p-0">
  <div class="m-0">
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

  <div class="p-0">
    {#if loading}
      <div class="table-wrap freeze-header">
        <MinimalTableSkeleton
          class="training-runs-table"
          columns={["Name", "Status", "Stage", "Model", "Dataset", "Recipe", "Group", "Tags", "Created", "Last updated", ""]}
          rows={8}
        />
      </div>
    {:else if error}
      <div class="page-empty">Failed to load: {error}</div>
    {:else if !allRuns.length}
      <div class="page-empty">No training runs found yet.</div>
    {:else if !filteredRuns.length}
      <div class="page-empty">No runs match the current filters.</div>
    {:else}
      <div class="table-wrap freeze-header">
        <ResizableTable class="training-runs-table" {columns} stickyFirstColumn>
          <tbody>
            {#each filteredRuns as run, runIndex (`${run.run_id || "run"}-${run.created_at || 0}-${runIndex}`)}
              {@const runName = run.run_id || "—"}
              {@const status = getStatus(run)}
              {@const stageLabel = frameworkStatusLabel(run)}
              {@const progress = frameworkProgress(run)}
              {@const groupTags = getGroupTags(run)}
              <tr class="run-row" class:row-selected={drawerRunId === run.run_id}>
                <td class="min-w-0 row-open-cell">
                  <a
                    href={trainingRunDetailPath(run.run_id)}
                    class="cell-open-button"
                    title={runName}
                    aria-label={`Open training run ${runName}`}
                    onclick={(event) => selectRun(run.run_id, event)}
                  >
                    <div class="block text-(--text-bright) [font-family:var(--font-mono)] [font-weight:400] text-[14px] leading-[20px] overflow-hidden text-ellipsis whitespace-nowrap">{runName}</div>
                  </a>
                </td>
                <td class="row-open-cell">
                  <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
                    <span class="flex flex-col items-start gap-[4px] min-w-0 max-w-full">
                      <StatusPill status={status} />
                      {#if resumeBadge(run)}
                        <span class="[border:1px_solid_color-mix(in_srgb,var(--yellow,#fbbf24)_42%,transparent)] rounded-[999px] bg-[color-mix(in_srgb,var(--yellow,#fbbf24)_10%,transparent)] text-(--yellow,#fbbf24) text-[11px] leading-[14px] p-[1px_6px] whitespace-nowrap">{resumeBadge(run)}</span>
                      {/if}
                    </span>
                  </a>
                </td>
                <td class="min-w-0 row-open-cell">
                  <a
                    href={trainingRunDetailPath(run.run_id)}
                    class="cell-open-button whitespace-normal! leading-[16px]!"
                    onclick={(event) => selectRun(run.run_id, event)}
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
                      <span class="text-(--muted)">—</span>
                    {/if}
                  </a>
                </td>
                <td class="min-w-0 row-open-cell" title={modelName(run)}>
                  <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
                    {modelName(run)}
                  </a>
                </td>
                <td class="min-w-0 row-open-cell" title={run.config_summary?.dataset_name || "—"}>
                  <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
                    {run.config_summary?.dataset_name || "—"}
                  </a>
                </td>
                <td class="row-open-cell">
                  <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
                    {run.framework || "—"}
                  </a>
                </td>
                <td class="group-cell row-open-cell" title={groupTags?.group_id || run.group_id || ""}>
                  <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
                    {#if groupTags?.group_id}
                      <span class="inline-block max-w-full whitespace-normal [overflow-wrap:anywhere] align-bottom p-[2px_8px] rounded-[999px] text-[0.72rem] [font-variant-numeric:tabular-nums] text-(--muted) [border:1px_solid_var(--border,#2f2f2f)] bg-[color-mix(in_srgb,var(--panel-alt)_70%,transparent)]">{groupTags.group_id}</span>
                    {:else}
                      <span class="group-empty">—</span>
                    {/if}
                  </a>
                </td>
                <td class="h-auto align-top row-open-cell">
                  <a
                    href={trainingRunDetailPath(run.run_id)}
                    class="cell-open-button overflow-visible! text-clip! whitespace-normal!"
                    onclick={(event) => selectRun(run.run_id, event)}
                  >
                    {#if groupTags?.tags.length}
                      <span class="flex flex-wrap gap-[4px] min-w-0 max-w-full">
                        {#each groupTags.tags as tag (tag.key)}
                          <span class="inline-flex items-baseline max-w-full min-w-0 overflow-hidden p-[2px_7px] [border:1px_solid_var(--border,#2f2f2f)] rounded-[999px] bg-[color-mix(in_srgb,var(--panel-alt)_70%,transparent)] text-(--text) [font-family:var(--font-mono)] text-[11px] leading-[14px]" title={`${tag.key}=${formatTagValue(tag.value)}`}>
                            <span class="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap text-(--muted)">{tag.key}</span><span>=</span><span class="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap">{formatTagValue(tag.value)}</span>
                          </span>
                        {/each}
                      </span>
                    {:else}
                      <span class="group-empty">—</span>
                    {/if}
                  </a>
                </td>
                <td class="whitespace-nowrap row-open-cell">
                  <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
                    <TimeAgo timestamp={run.started_at || run.created_at} showJustNow falsyRepresentation="—" />
                  </a>
                </td>
                <td class="whitespace-nowrap row-open-cell">
                  <a href={trainingRunDetailPath(run.run_id)} class="cell-open-button" onclick={(event) => selectRun(run.run_id, event)}>
                    <TimeAgo timestamp={run.updated_at} showJustNow falsyRepresentation="—" />
                  </a>
                </td>
                <td class="min-w-0 overflow-visible">
                  <div class="flex items-center flex-wrap gap-[6px]">
                    <button
                      class="inline-flex items-center gap-[6px] whitespace-nowrap [border:1px_solid_var(--color-c-gray-10,#2f2f2f)] rounded-[6px] p-[4px_8px] [font:inherit] text-[12px] font-medium leading-[16px] text-(--muted) bg-transparent cursor-pointer ghost-hover"
                      title="Open expanded view"
                      aria-label={`Open expanded view for training run ${run.run_id}`}
                      onclick={(event) => {
                        event.stopPropagation();
                        selectRun(run.run_id, event);
                      }}
                    >
                      <Maximize2 size={12} strokeWidth={2.1} />
                      <span class="expand-button-label">Expand</span>
                    </button>
                    {#if run.modal_app_url}
                      <a
                        class="training-open-modal-link"
                        href={run.modal_app_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        onclick={(event) => event.stopPropagation()}
                      >
                        <span class="open-modal-link-label">Open in Modal</span>
                        <ExternalLink class="training-open-modal-link-icon" size={12} strokeWidth={2.1} />
                      </a>
                    {:else}
                      <span class="training-open-modal-link open-modal-link-disabled">
                        <span class="open-modal-link-label">Open in Modal</span>
                        <ExternalLink class="training-open-modal-link-icon" size={12} strokeWidth={2.1} />
                      </span>
                    {/if}
                    {#each run.wandb_links || [] as link (link.url)}
                      <a
                        class="training-open-modal-link training-open-wandb-link"
                        href={link.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        onclick={(event) => event.stopPropagation()}
                      >
                        <span class="open-modal-link-label">{link.label}</span>
                        <ExternalLink class="training-open-modal-link-icon" size={12} strokeWidth={2.1} />
                      </a>
                    {/each}
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
      class="w-full h-full max-h-[100vh]"
      aria-label={`Training run ${selectedRun.run_id}`}
    >
      <div class="drawer-panel-header">
        <div class="min-w-0 overflow-hidden">
          <div class="drawer-panel-eyebrow">Training run</div>
          <h2 class="text-(--text-bright) text-[16px] font-medium [font-family:var(--font-mono)] leading-[24px] whitespace-nowrap overflow-hidden text-ellipsis" title={selectedRun.run_id}>{selectedRun.run_id}</h2>
        </div>
        <div class="flex items-center gap-[8px] shrink-0">
          <button
            class="inline-flex items-center gap-[6px] [border:1px_solid_var(--color-c-gray-10,#2f2f2f)] rounded-[6px] p-[4px_8px] [font:inherit] text-[12px] font-medium leading-[16px] text-(--muted) bg-transparent cursor-pointer ghost-hover"
            onclick={(event) => selectRun(selectedRun.run_id, event)}
            title="Expand to full view"
          >
            <Maximize2 size={12} />
            <span>Expand</span>
          </button>
          {#if selectedRun.modal_app_url}
            <a
              class="inline-flex items-center gap-[6px] [border:1px_solid_var(--color-c-gray-10,#2f2f2f)] rounded-[6px] p-[4px_8px] no-underline text-(--muted) text-[12px] font-medium leading-[16px] ghost-hover"
              href={selectedRun.modal_app_url}
              target="_blank"
              rel="noopener noreferrer"
            >
              <span>View in app</span>
              <ExternalLink size={12} />
            </a>
          {/if}
          <button class="drawer-panel-close ghost-hover" onclick={closeDrawer} aria-label="Close run drawer">
            <PanelRightClose size={16} />
          </button>
        </div>
      </div>

      <div class="p-[4px_20px_16px]">
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

