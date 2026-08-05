<script>
  import { onMount } from "svelte";
  import { Book, Zap } from "lucide-svelte";
  import "./app.css";
  import Sidebar from "./components/Sidebar.svelte";
  import DashboardHeader from "./components/DashboardHeader.svelte";
  import TrainingPage from "./pages/TrainingPage.svelte";
  import TrainingRunDetailPage from "./pages/TrainingRunDetailPage.svelte";
  import { fetchRuns } from "./lib/api.js";
  import logoSvg from "./lib/logo.svg";
  import { fmtDuration } from "./lib/format.js";

  const DOCS_URL = "https://gym.modal.dev";

  let allRuns = $state([]);
  let loading = $state(true);
  let error = $state(null);
  let search = $state("");
  let activeRecipes = $state(new Set());
  let activeStatuses = $state(new Set());
  let activeGroups = $state(new Set());
  let trainingGroupBy = $state("none");
  // Recipe/status/group values we've seen across loads. New ones are
  // auto-enabled in the filters once; the user's selections are never reset by
  // a refresh.
  let seenRecipes = new Set();
  let seenStatuses = new Set();
  let seenGroups = new Set();
  let activePage = $state("training");
  let activeTrainingRunId = $state(null);
  // When set (and no full detail page is open), the training list shows a
  // summary drawer for this run — set by "Collapse" on the detail page.
  let drawerRunId = $state(null);
  // True while any data fetch is in flight (manual or the 5s auto-refresh) —
  // drives the spinning refresh button. Distinct from `loading`, which only
  // gates the cold-start skeleton.
  let refreshing = $state(false);
  let runsRequestId = 0;
  let hasLoadedRuns = false;
  let initialRunsLoadStarted = false;

  const pageMeta = {
    training: { title: "Training runs" },
  };

  const pagePaths = {
    training: "/training",
  };

  function pageFromPath(pathname) {
    if (pathname === "/" || pathname.startsWith("/training")) return "training";
    return "training";
  }

  function runIdFromPath(pathname) {
    if (!pathname.startsWith("/training/")) return null;
    const tail = pathname.slice("/training/".length).split("/")[0];
    return tail ? decodeURIComponent(tail) : null;
  }

  const navItems = [
    { key: "training", label: "Training runs", Icon: Zap, path: pagePaths.training },
  ];

  if (typeof window !== "undefined") {
    activePage = pageFromPath(window.location.pathname);
    activeTrainingRunId = runIdFromPath(window.location.pathname);
  }

  onMount(() => {
    const syncPageWithPath = () => {
      activePage = pageFromPath(window.location.pathname);
      activeTrainingRunId = runIdFromPath(window.location.pathname);
    };

    if (window.location.pathname === "/") {
      window.history.replaceState({}, "", pagePaths.training);
    } else {
      syncPageWithPath();
    }

    window.addEventListener("popstate", syncPageWithPath);

    // Auto-refresh the active page's data every 5s so running training runs,
    // their status/stage and rollouts stay live. Current data stays on screen
    // (no skeleton) and only the refresh button spins while fetching. A run
    // detail page refreshes its own run, so skip the full list there.
    const refresh = window.setInterval(() => {
      if (activePage === "training" && activeTrainingRunId) return;
      void load();
    }, 5000);

    return () => {
      window.removeEventListener("popstate", syncPageWithPath);
      window.clearInterval(refresh);
    };
  });

  function getRecipe(run) {
    return run.recipe || run.framework || "(untagged)";
  }

  const NO_GROUP = "(no group)";

  function getGroup(run) {
    return safeText(run.group_id) || NO_GROUP;
  }

  function getStatus(run) {
    return safeText(run.display_status) || "pending";
  }

  function getTrainingRunStatus(run) {
    return safeText(run.status).toLowerCase();
  }

  function getFrameworkStatus(run) {
    return safeText(run.framework_status);
  }

  function showFrameworkStatus(run) {
    if (getTrainingRunStatus(run) === "running") return true;
    return !!run.framework_status;
  }

  function modelName(run) {
    return run.model || "—";
  }

  function safeText(value) {
    if (value && typeof value === "object" && "value" in value) return value.value;
    return value != null ? String(value) : "";
  }

  function includesText(value, query) {
    return safeText(value).toLowerCase().includes(query);
  }

  function getErrorMessage(value) {
    if (value instanceof Error) return value.message;
    if (typeof value === "string") return value;
    return "unknown error";
  }

  function fetchWithTimeout(fn, timeoutMs, label) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    return fn({ signal: controller.signal })
      .then((value) => {
        clearTimeout(timeoutId);
        return value;
      })
      .catch((err) => {
        clearTimeout(timeoutId);
        if (err.name === "AbortError")
          throw new Error(`${label} request timed out after ${timeoutMs}ms`);
        throw err;
      });
  }

  async function loadRuns() {
    const requestId = ++runsRequestId;
    const isStale = () => requestId !== runsRequestId;

    // Skeleton only until the first response settles. Once we've completed a
    // load attempt (success or failure), refreshes keep the current rows on
    // screen — the spinning refresh button is the only "loading" affordance.
    if (!hasLoadedRuns) loading = true;
    error = null;

    try {
      const runs = await fetchWithTimeout(fetchRuns, 30000, "runs");
      if (isStale()) return;
      allRuns = runs;
      // Auto-enable newly-seen recipes/statuses without resetting the user's
      // current filter selection on every refresh.
      const nextRecipes = new Set(activeRecipes);
      const nextStatuses = new Set(activeStatuses);
      const nextGroups = new Set(activeGroups);
      let recipesChanged = false;
      let statusesChanged = false;
      let groupsChanged = false;
      for (const run of allRuns) {
        const recipe = getRecipe(run);
        if (!seenRecipes.has(recipe)) {
          seenRecipes.add(recipe);
          nextRecipes.add(recipe);
          recipesChanged = true;
        }
        const status = getStatus(run);
        if (!seenStatuses.has(status)) {
          seenStatuses.add(status);
          nextStatuses.add(status);
          statusesChanged = true;
        }
        const group = getGroup(run);
        if (!seenGroups.has(group)) {
          seenGroups.add(group);
          nextGroups.add(group);
          groupsChanged = true;
        }
      }
      if (recipesChanged) activeRecipes = nextRecipes;
      if (statusesChanged) activeStatuses = nextStatuses;
      if (groupsChanged) activeGroups = nextGroups;
    } catch (e) {
      if (isStale()) return;
      // Keep the data we already have on a transient refresh failure — only
      // surface the error (and clear) when there's nothing to show yet.
      // Otherwise the page flickers to "Loading…"/empty on every flaky poll.
      if (!allRuns.length) {
        error = getErrorMessage(e);
        activeRecipes = new Set();
        activeStatuses = new Set();
        activeGroups = new Set();
      }
    } finally {
      // Always retire the cold-start skeleton once any attempt settles — even a
      // stale one. A slow request superseded by the 5s auto-refresh must not
      // leave `loading` pinned true forever.
      hasLoadedRuns = true;
      loading = false;
    }
  }

  async function load() {
    refreshing = true;
    try {
      const tasks = [loadRuns()];
      await Promise.all(tasks);
    } finally {
      refreshing = false;
    }
  }

  $effect(() => {
    if (
      !activeTrainingRunId &&
      !hasLoadedRuns &&
      !initialRunsLoadStarted
    ) {
      initialRunsLoadStarted = true;
      void loadRuns();
    } else if (activeTrainingRunId && !hasLoadedRuns) {
      loading = false;
    }
  });

  let recipes = $derived([...new Set(allRuns.map(getRecipe))].sort());
  let statuses = $derived([...new Set(allRuns.map(getStatus))].sort());
  // Real group ids first (alphabetical), with "(no group)" pinned last so the
  // sweep groups are what you see at the top of the filter.
  let groups = $derived(
    [...new Set(allRuns.map(getGroup))].sort((a, b) => {
      if (a === NO_GROUP) return 1;
      if (b === NO_GROUP) return -1;
      return a.localeCompare(b);
    }),
  );

  let recipeCounts = $derived(
    allRuns.reduce((acc, run) => {
      const recipe = getRecipe(run);
      acc[recipe] = (acc[recipe] || 0) + 1;
      return acc;
    }, {}),
  );

  let statusCounts = $derived(
    allRuns.reduce((acc, run) => {
      const status = getStatus(run);
      acc[status] = (acc[status] || 0) + 1;
      return acc;
    }, {}),
  );

  let groupCounts = $derived(
    allRuns.reduce((acc, run) => {
      const group = getGroup(run);
      acc[group] = (acc[group] || 0) + 1;
      return acc;
    }, {}),
  );

  let filteredRuns = $derived(
    allRuns
      .filter((run) => {
        if (!activeRecipes.has(getRecipe(run))) return false;
        if (!activeStatuses.has(getStatus(run))) return false;
        if (!activeGroups.has(getGroup(run))) return false;
        if (search) {
          const q = search.toLowerCase();
          if (
            !includesText(run.run_id, q) &&
            !includesText(run.modal_app_id, q) &&
            !includesText(run.group_id, q) &&
            !includesText(JSON.stringify(run.group_tags || {}), q) &&
            !includesText(run.model, q) &&
            !includesText(run.dataset, q) &&
            !includesText(run.train_result?.training_run_id, q) &&
            !includesText(run.train_result?.checkpoint_dir, q) &&
            !includesText(run.train_result?.model_name, q) &&
            !includesText(run.train_result?.model_path, q) &&
            !includesText(run.framework_status, q) &&
            !includesText(run.deployment_id, q)
          ) {
            return false;
          }
        }
        return true;
      })
      .sort((a, b) => (b.created_at || 0) - (a.created_at || 0)),
  );

  const trainingGroupKeyFns = {
    group: getGroup,
    dataset: (run) => safeText(run.dataset) || "(no dataset)",
    model: modelName,
  };

  const trainingGroupKey = (run, groupBy) => trainingGroupKeyFns[groupBy]?.(run) ?? "";

  // Buckets inherit filteredRuns' recency sort: groups come out ordered by
  // newest member and runs stay sorted within each group.
  let trainingRunGroups = $derived.by(() => {
    if (trainingGroupBy === "none") return [];
    const buckets = Map.groupBy(filteredRuns, (run) => trainingGroupKey(run, trainingGroupBy));
    return [...buckets].map(([key, runs]) => ({
      key,
      runs,
      latestCreatedAt: runs[0]?.created_at || null,
    }));
  });

  let completedTotal = $derived(allRuns.filter((run) => getStatus(run) === "completed").length);
  let cancelledTotal = $derived(allRuns.filter((run) => getStatus(run) === "cancelled").length);
  let stoppedTotal = $derived(allRuns.filter((run) => getStatus(run) === "stopped").length);
  let failedTotal = $derived(allRuns.filter((run) => getStatus(run) === "failed").length);
  let runningTotal = $derived(
    allRuns.length - completedTotal - cancelledTotal - stoppedTotal - failedTotal,
  );

  let activeTrainingRun = $derived(
    allRuns.find((run) => run.run_id === activeTrainingRunId) || null,
  );

  let statusText = $derived.by(() => {
    if (activePage === "training" && activeTrainingRunId) return "run details";
    if (activePage === "training" && loading) return "loading...";
    if (error) return "error";
    if (!allRuns.length) return "0 runs";
    return `${filteredRuns.length} of ${allRuns.length} runs`;
  });

  function toggleRecipe(recipe) {
    const next = new Set(activeRecipes);
    if (next.has(recipe)) next.delete(recipe);
    else next.add(recipe);
    activeRecipes = next;
  }

  function selectAllRecipes() {
    activeRecipes = new Set(recipes);
  }

  function clearRecipes() {
    activeRecipes = new Set();
  }

  function toggleStatus(status) {
    const next = new Set(activeStatuses);
    if (next.has(status)) next.delete(status);
    else next.add(status);
    activeStatuses = next;
  }

  function selectAllStatuses() {
    activeStatuses = new Set(statuses);
  }

  function clearStatuses() {
    activeStatuses = new Set();
  }

  function toggleGroup(group) {
    const next = new Set(activeGroups);
    if (next.has(group)) next.delete(group);
    else next.add(group);
    activeGroups = next;
  }

  function selectAllGroups() {
    activeGroups = new Set(groups);
  }

  function clearGroups() {
    activeGroups = new Set();
  }

  function setActivePage(page) {
    activePage = page;
    activeTrainingRunId = null;
    drawerRunId = null;
    if (typeof window === "undefined") return;
    const targetPath = pagePaths[page] || pagePaths.training;
    if (window.location.pathname !== targetPath) {
      window.history.pushState({}, "", targetPath);
    }
  }

  function backToTrainingList() {
    activeTrainingRunId = null;
    drawerRunId = null;
    if (typeof window === "undefined") return;
    if (window.location.pathname !== pagePaths.training) {
      window.history.pushState({}, "", pagePaths.training);
    }
  }

  // Opening a run shows the full detail page (a real route, not a drawer).
  function openTrainingRunDetail(runId) {
    drawerRunId = null;
    activeTrainingRunId = runId;
    if (typeof window === "undefined") return;
    const target = `${pagePaths.training}/${encodeURIComponent(runId)}`;
    if (window.location.pathname !== target) {
      window.history.pushState({}, "", target);
    }
  }

  // "Collapse" on the detail page drops back to the list and reopens the run
  // as a summary drawer.
  function collapseTrainingRunToDrawer() {
    drawerRunId = activeTrainingRunId;
    activeTrainingRunId = null;
    if (typeof window === "undefined") return;
    if (window.location.pathname !== pagePaths.training) {
      window.history.pushState({}, "", pagePaths.training);
    }
  }

  function closeTrainingDrawer() {
    drawerRunId = null;
  }
</script>

<div class="h-[100dvh] grid grid-rows-[auto_1fr] bg-(--bg) overflow-x-hidden">
  <header class="[border-bottom:1px_solid_var(--color-c-surface-highlight-gray-opaque,#272727)] bg-(--bg-depth) flex items-center justify-between gap-[1rem] min-h-[53px] p-[0_1rem] max-[900px]:min-h-[53px] max-[900px]:p-[0_0.75rem]">
    <div class="inline-flex items-center gap-[0.55rem] flex-[0_0_auto] min-w-0">
      <img src={logoSvg} alt="Modal" class="h-[17.5px] w-auto flex-[0_0_auto]" />
      <span class="inline-flex items-center gap-[0.18rem] [font-family:var(--font-display)] [font-feature-settings:'ss01'_on] text-[17.6px] leading-[1] [padding-block:0.08rem] font-[600] tracking-[-0.02em] [transform:translateY(1px)] whitespace-nowrap max-[360px]:text-[15px]">
        <span class="text-[#ddffdc]">Modal</span>
        <span class="text-(--green)">Training Gym</span>
      </span>
    </div>
    <a
      class="[border:0] rounded-[10px] text-(--text) [background:transparent] [text-decoration:none] text-[14px] font-medium p-[8px] inline-flex items-center gap-[8px] flex-[0_0_auto] hover:text-(--text-bright) hover:[background:color-mix(in_srgb,white_4%,transparent)] max-[520px]:hidden"
      href={DOCS_URL}
      target="_blank"
      rel="noopener noreferrer"
    >
      <Book size={14} strokeWidth={2.1} />
      <span>Docs</span>
    </a>
  </header>

  <div class="grid grid-cols-[232px_minmax(0,1fr)] min-h-0 h-full bg-(--bg) max-[900px]:grid-cols-[1fr] max-[900px]:grid-rows-[auto_minmax(0,1fr)]">
    <Sidebar {navItems} {activePage} onNavigate={setActivePage} />

    <main class="min-w-0 min-h-0 h-full flex flex-col overflow-y-auto">
      <DashboardHeader
        title={pageMeta[activePage].title}
        {statusText}
        {refreshing}
        onRefresh={load}
      />

    {#if activePage === "training" && activeTrainingRunId}
      <TrainingRunDetailPage
        runId={activeTrainingRunId}
        initialRun={activeTrainingRun}
        {modelName}
        {getStatus}
        {getFrameworkStatus}
        {showFrameworkStatus}
        {fmtDuration}
        onBack={backToTrainingList}
        onCollapse={collapseTrainingRunToDrawer}
      />
    {:else if activePage === "training"}
      <TrainingPage
        {allRuns}
        {completedTotal}
        {runningTotal}
        {stoppedTotal}
        {failedTotal}
        {recipes}
        {recipeCounts}
        {activeRecipes}
        {statuses}
        {statusCounts}
        {activeStatuses}
        {groups}
        {groupCounts}
        {activeGroups}
        {filteredRuns}
        runGroups={trainingRunGroups}
        bind:groupBy={trainingGroupBy}
        {loading}
        {error}
        {modelName}
        {getStatus}
        {showFrameworkStatus}
        {fmtDuration}
        bind:search
        {drawerRunId}
        onOpenDetail={openTrainingRunDetail}
        onCloseDrawer={closeTrainingDrawer}
        onToggleRecipe={toggleRecipe}
        onSelectAllRecipes={selectAllRecipes}
        onClearRecipes={clearRecipes}
        onToggleStatus={toggleStatus}
        onSelectAllStatuses={selectAllStatuses}
        onClearStatuses={clearStatuses}
        onToggleGroup={toggleGroup}
        onSelectAllGroups={selectAllGroups}
        onClearGroups={clearGroups}
      />
    {/if}
    </main>
  </div>
</div>
