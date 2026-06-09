<script>
  import { ArrowLeft, ExternalLink } from "lucide-svelte";
  import FrameworkStageProgress from "../components/FrameworkStageProgress.svelte";
  import StatusPill from "../components/StatusPill.svelte";
  import TimeAgo from "../components/TimeAgo.svelte";
  import { fetchRunRollouts, fetchRollout } from "../lib/api.js";
  import { smoothedStageLabel } from "../lib/format.js";

  let {
    runId,
    allRuns,
    modelName,
    getStatus,
    getFrameworkStatus,
    showFrameworkStatus,
    fmtDuration,
    onBack,
    // When rendered inside the expanded run drawer the surrounding UI already
    // shows the header/title/summary, so we hide them and render only the
    // unique rollouts + logs content.
    embedded = false,
  } = $props();

  let run = $derived.by(() =>
    (allRuns || []).find((r) => r.run_id === runId) || null
  );

  function frameworkProgress() {
    const p = run?.framework_progress;
    if (!p || typeof p !== "object") return null;
    const current = Number(p.current);
    const total = Number(p.total);
    if (!Number.isFinite(current) || !Number.isFinite(total) || total <= 0) {
      return null;
    }
    return {
      current: Math.max(0, Math.min(current, total)),
      total,
      unit: p.unit || "step",
    };
  }

  function frameworkStatusLabel() {
    return smoothedStageLabel(
      getFrameworkStatus?.(run),
      run?.framework_progress,
    );
  }

  function progressLabel(progress) {
    if (!progress) return "";
    const unit = String(progress.unit || "step");
    const label = unit.charAt(0).toUpperCase() + unit.slice(1);
    return `${label} ${progress.current} / ${progress.total}`;
  }

  function runDuration() {
    if (!run) return "—";
    if (typeof run.duration_seconds === "number" && run.duration_seconds >= 0) {
      return fmtDuration(0, run.duration_seconds);
    }
    if (run.started_at) {
      return fmtDuration(run.started_at, run.ended_at);
    }
    return "—";
  }

  function formatMean(value) {
    if (typeof value !== "number" || !Number.isFinite(value)) return "—";
    return value.toFixed(3);
  }

  // ── Rollouts (auto-refresh while run is running) ─────────────────────
  let rolloutSummaries = $state([]);
  let rolloutsLoading = $state(false);
  let rolloutsError = $state("");
  let expandedRolloutId = $state(null);
  let expandedRollout = $state(null);
  let expandedRolloutLoading = $state(false);

  async function loadRollouts(signal) {
    if (!runId) return;
    try {
      const rows = await fetchRunRollouts(runId, { signal });
      if (signal?.aborted) return;
      rolloutSummaries = rows;
      rolloutsError = "";
    } catch (err) {
      if (signal?.aborted) return;
      rolloutsError = String(err?.message || err);
    } finally {
      rolloutsLoading = false;
    }
  }

  $effect(() => {
    const id = runId;
    rolloutSummaries = [];
    rolloutsError = "";
    expandedRolloutId = null;
    expandedRollout = null;
    if (!id) return;

    const controller = new AbortController();
    rolloutsLoading = true;
    void loadRollouts(controller.signal);

    // Poll while the run is active so new rollouts stream in.
    const interval = window.setInterval(() => {
      const status = String(run?.status || "").toLowerCase();
      if (status && status !== "running") return;
      void loadRollouts(controller.signal);
    }, 8000);

    return () => {
      controller.abort();
      window.clearInterval(interval);
    };
  });

  async function toggleRolloutDetail(rolloutId) {
    if (!runId) return;
    if (expandedRolloutId === rolloutId) {
      expandedRolloutId = null;
      expandedRollout = null;
      return;
    }
    expandedRolloutId = rolloutId;
    expandedRollout = null;
    expandedRolloutLoading = true;
    try {
      const detail = await fetchRollout(runId, rolloutId);
      if (expandedRolloutId === rolloutId) {
        expandedRollout = detail;
      }
    } finally {
      if (expandedRolloutId === rolloutId) {
        expandedRolloutLoading = false;
      }
    }
  }

  // ── Live Modal log stream (SSE, pure pass-through) ───────────────────
  const LOG_BUFFER_MAX = 2000;
  let logLines = $state([]); // [{task_id, line, ts}]
  let logState = $state("idle"); // idle | streaming | paused | done | error | reconnecting
  let logError = $state("");
  let logDropped = $state(0); // server-side rate-capped lines (cumulative since reconnect)
  let logTailEl = $state(null);

  // User controls
  let logPaused = $state(false);
  let logSearch = $state("");
  let logSearchInput = $state(""); // debounced into logSearch
  let logRateCap = $state(0); // 0 = no cap
  let logFollow = $state(true); // auto-scroll to bottom

  // Debounce search input → URL
  $effect(() => {
    const value = logSearchInput;
    const handle = window.setTimeout(() => {
      logSearch = value.trim();
    }, 350);
    return () => window.clearTimeout(handle);
  });

  $effect(() => {
    const id = runId;
    const status = String(run?.status || "").toLowerCase();
    // Re-create the EventSource whenever any of the connection params change.
    const search = logSearch;
    const rate = logRateCap;
    const paused = logPaused;

    logLines = [];
    logState = "idle";
    logError = "";
    logDropped = 0;
    if (!id || status !== "running" || paused) {
      if (paused) logState = "paused";
      return;
    }

    const params = new URLSearchParams();
    if (search) params.set("search", search);
    if (rate > 0) params.set("max_lines_per_sec", String(rate));
    const qs = params.toString();
    const url =
      `/api/runs/${encodeURIComponent(id)}/logs/stream` +
      (qs ? `?${qs}` : "");

    const es = new EventSource(url);
    logState = "streaming";

    es.onmessage = (evt) => {
      try {
        const payload = JSON.parse(evt.data);
        const line = String(payload.line || "");
        if (!line) return;
        const parts = line.split(/\r?\n/);
        const additions = parts
          .filter((p) => p.length > 0)
          .map((p) => ({
            task_id: payload.task_id || "",
            line: p,
            ts: payload.ts || Date.now(),
          }));
        if (!additions.length) return;
        const next = [...logLines, ...additions];
        logLines =
          next.length > LOG_BUFFER_MAX ? next.slice(-LOG_BUFFER_MAX) : next;
        if (logFollow && logTailEl) {
          queueMicrotask(() => {
            if (logTailEl) logTailEl.scrollTop = logTailEl.scrollHeight;
          });
        }
      } catch {
        // ignore malformed payloads
      }
    };

    es.addEventListener("done", () => {
      logState = "done";
      es.close();
    });

    es.addEventListener("reconnect", (evt) => {
      try {
        const { reason } = JSON.parse(evt.data || "{}");
        logError = String(reason || "");
      } catch {
        logError = "";
      }
      logState = "reconnecting";
    });

    es.addEventListener("dropped", (evt) => {
      try {
        const { dropped } = JSON.parse(evt.data || "{}");
        logDropped += Number(dropped) || 0;
      } catch {}
    });

    es.addEventListener("error", (evt) => {
      try {
        const { error } = JSON.parse(evt.data || "{}");
        logError = String(error || "");
      } catch {
        logError = "";
      }
      logState = "error";
      es.close();
    });

    es.onerror = () => {
      if (logState === "streaming") logState = "reconnecting";
    };

    return () => {
      try {
        es.close();
      } catch {}
    };
  });

  function toggleLogPaused() {
    logPaused = !logPaused;
  }

  function clearLogs() {
    logLines = [];
    logDropped = 0;
  }

  let chartPath = $derived.by(() => {
    if (!rolloutSummaries.length) return "";
    const points = rolloutSummaries.map((r) => ({
      x: Number(r.rollout_id) || 0,
      y: Number(r.mean) || 0,
    }));
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const yMin = Math.min(...ys);
    const yMax = Math.max(...ys);
    const xSpan = xMax - xMin || 1;
    const ySpan = yMax - yMin || 1;
    const W = 640;
    const H = 140;
    return points
      .map((p, i) => {
        const x = ((p.x - xMin) / xSpan) * W;
        const y = H - ((p.y - yMin) / ySpan) * (H - 4) - 2;
        return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");
  });

  let chartStats = $derived.by(() => {
    if (!rolloutSummaries.length) return null;
    const means = rolloutSummaries.map((r) => Number(r.mean) || 0);
    const min = Math.min(...means);
    const max = Math.max(...means);
    const latest = means[means.length - 1];
    return { min, max, latest };
  });
</script>

<section class="detail" class:embedded>
  {#if !embedded}
    <header class="detail-header">
      <button class="back-button" onclick={onBack}>
        <ArrowLeft size={14} strokeWidth={2.1} />
        <span>Back to runs</span>
      </button>
      {#if run?.modal_app_url}
        <a
          class="modal-link"
          href={run.modal_app_url}
          target="_blank"
          rel="noopener noreferrer"
        >
          <span>Open in Modal</span>
          <ExternalLink size={12} strokeWidth={2.1} />
        </a>
      {/if}
    </header>
  {/if}

  {#if !run}
    <div class="empty">Loading run {runId}…</div>
  {:else}
    {#if !embedded}
    <div class="detail-title-row">
      <h1 class="detail-title" title={run.run_id}>{run.run_id}</h1>
      <StatusPill status={getStatus(run)} />
    </div>

    <dl class="detail-meta">
      <div>
        <dt>Model</dt>
        <dd>{modelName(run)}</dd>
      </div>
      <div>
        <dt>Dataset</dt>
        <dd>{run.config_summary?.dataset_name || "—"}</dd>
      </div>
      <div>
        <dt>Framework</dt>
        <dd>{run.framework || "—"}</dd>
      </div>
      <div>
        <dt>Started</dt>
        <dd>
          <TimeAgo timestamp={run.started_at || run.created_at} showJustNow />
        </dd>
      </div>
      <div>
        <dt>Last updated</dt>
        <dd>
          <TimeAgo timestamp={run.updated_at} showJustNow falsyRepresentation="—" />
        </dd>
      </div>
      <div>
        <dt>Duration</dt>
        <dd>{runDuration()}</dd>
      </div>
      {#if showFrameworkStatus(run) && frameworkStatusLabel()}
        <div class="detail-stage">
          <dt>Stage</dt>
          <dd>
            <FrameworkStageProgress
              progress={frameworkProgress()}
              progressLabel={progressLabel(frameworkProgress())}
              stageLabel={frameworkStatusLabel()}
            />
          </dd>
        </div>
      {/if}
    </dl>
    {/if}

    <section class="rollouts">
      <div class="rollouts-header">
        <h2>Rollouts</h2>
        <span class="rollouts-count">
          {rolloutSummaries.length} step{rolloutSummaries.length === 1 ? "" : "s"}
        </span>
      </div>

      {#if rolloutsLoading && !rolloutSummaries.length}
        <div class="empty">Loading rollouts…</div>
      {:else if rolloutsError}
        <div class="empty">Failed to load rollouts: {rolloutsError}</div>
      {:else if !rolloutSummaries.length}
        <div class="empty">No rollouts recorded yet.</div>
      {:else}
        <div class="rollout-chart">
          <svg viewBox="0 0 640 140" preserveAspectRatio="none" aria-hidden="true">
            <path d={chartPath} fill="none" stroke="var(--accent)" stroke-width="1.5" />
          </svg>
          {#if chartStats}
            <div class="rollout-chart-meta">
              <span>min {formatMean(chartStats.min)}</span>
              <span>latest {formatMean(chartStats.latest)}</span>
              <span>max {formatMean(chartStats.max)}</span>
            </div>
          {/if}
        </div>

        <table class="rollout-table">
          <thead>
            <tr>
              <th>Step</th>
              <th>Mean reward</th>
              <th>Samples</th>
              <th>When</th>
            </tr>
          </thead>
          <tbody>
            {#each rolloutSummaries as r (r.rollout_id)}
              <tr
                class:expanded={expandedRolloutId === r.rollout_id}
                onclick={() => toggleRolloutDetail(r.rollout_id)}
              >
                <td>#{r.rollout_id}</td>
                <td class="rollout-mean">{formatMean(r.mean)}</td>
                <td>{r.total}</td>
                <td>
                  <TimeAgo timestamp={r.created_at} showJustNow falsyRepresentation="—" />
                </td>
              </tr>
              {#if expandedRolloutId === r.rollout_id}
                <tr class="rollout-detail-row">
                  <td colspan="4">
                    {#if expandedRolloutLoading}
                      <div class="empty">Loading samples…</div>
                    {:else if !expandedRollout}
                      <div class="empty">No samples recorded.</div>
                    {:else}
                      {#each expandedRollout.samples || [] as sample, i (i)}
                        <div class="rollout-sample">
                          <div class="rollout-sample-header">
                            <span>Sample {i + 1}</span>
                            <span class="rollout-sample-score">
                              reward {formatMean(sample.score)}
                            </span>
                          </div>
                          {#if sample.prompt}
                            <div class="rollout-sample-label">prompt</div>
                            <pre class="rollout-sample-text">{sample.prompt}</pre>
                          {/if}
                          {#if sample.response}
                            <div class="rollout-sample-label">response</div>
                            <pre class="rollout-sample-text">{sample.response}</pre>
                          {/if}
                        </div>
                      {/each}
                    {/if}
                  </td>
                </tr>
              {/if}
            {/each}
          </tbody>
        </table>
      {/if}
    </section>

    <section class="logs">
      <div class="logs-header">
        <h2>Modal logs</h2>
        <span class="logs-status">
          {#if logState === "streaming"}
            <span class="dot dot-live"></span> live
          {:else if logState === "paused"}
            <span class="dot dot-dim"></span> paused
          {:else if logState === "reconnecting"}
            <span class="dot dot-warn"></span> reconnecting…
          {:else if logState === "done"}
            <span class="dot dot-dim"></span> finished
          {:else if logState === "error"}
            <span class="dot dot-err"></span> error
          {:else if String(run?.status || "").toLowerCase() !== "running"}
            <span class="dot dot-dim"></span> run not active
          {:else}
            <span class="dot dot-dim"></span> idle
          {/if}
        </span>
      </div>

      <div class="logs-controls">
        <button
          class="log-button"
          onclick={toggleLogPaused}
          disabled={String(run?.status || "").toLowerCase() !== "running"}
        >
          {logPaused ? "Resume" : "Pause"}
        </button>
        <button class="log-button" onclick={clearLogs} disabled={!logLines.length}>
          Clear
        </button>
        <input
          class="log-search"
          type="search"
          placeholder="filter substring…"
          bind:value={logSearchInput}
          aria-label="Filter log lines"
        />
        <label class="log-rate">
          <span>Rate cap</span>
          <select bind:value={logRateCap} aria-label="Lines per second cap">
            <option value={0}>off</option>
            <option value={10}>10/s</option>
            <option value={50}>50/s</option>
            <option value={200}>200/s</option>
            <option value={1000}>1000/s</option>
          </select>
        </label>
        <label class="log-follow">
          <input type="checkbox" bind:checked={logFollow} />
          <span>Follow tail</span>
        </label>
      </div>

      {#if logState === "error" && logError}
        <div class="empty">Log stream error: {logError}</div>
      {/if}

      {#if !logLines.length}
        <div class="empty">
          {#if String(run?.status || "").toLowerCase() !== "running"}
            Logs only stream while the run is active.
          {:else if logPaused}
            Stream paused.
          {:else if logSearch}
            Waiting for log output matching "{logSearch}"…
          {:else}
            Waiting for log output…
          {/if}
        </div>
      {:else}
        <div class="log-tail" bind:this={logTailEl}>
          {#each logLines as entry, i (i)}
            <div class="log-row">
              <span class="log-task">{entry.task_id || ""}</span>
              <span class="log-line">{entry.line}</span>
            </div>
          {/each}
        </div>
        <div class="log-meta">
          <span>
            Showing last {logLines.length} line{logLines.length === 1 ? "" : "s"} (cap {LOG_BUFFER_MAX})
          </span>
          {#if logDropped > 0}
            <span class="log-meta-drop">
              · {logDropped} dropped by rate cap
            </span>
          {/if}
        </div>
      {/if}
    </section>
  {/if}
</section>

<style>
  .detail {
    padding: 24px 32px 64px;
    max-width: 980px;
    margin: 0 auto;
    color: var(--text);
  }

  /* Inside the expanded drawer the drawer owns padding/width, so drop the
     page chrome and let the rollouts + logs fill the wide drawer. */
  .detail.embedded {
    padding: 0;
    max-width: none;
    margin: 0;
  }

  .detail-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
  }

  .back-button {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: none;
    border: 0;
    color: var(--muted);
    cursor: pointer;
    font-size: 13px;
    padding: 4px 8px;
    border-radius: 6px;
  }

  .back-button:hover {
    color: var(--text);
    background: var(--color-c-gray-10, #2f2f2f);
  }

  .modal-link {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--muted);
    font-size: 12px;
    text-decoration: none;
  }

  .modal-link:hover {
    color: var(--accent);
  }

  .detail-title-row {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 24px;
  }

  .detail-title {
    font-size: 22px;
    font-weight: 600;
    color: var(--text-bright);
    margin: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .detail-meta {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px 24px;
    margin: 0 0 32px;
    padding: 0;
  }

  .detail-meta > div {
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
  }

  .detail-meta dt {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
  }

  .detail-meta dd {
    margin: 0;
    font-size: 13px;
    color: var(--text-bright);
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .detail-stage {
    grid-column: 1 / -1;
  }

  .rollouts {
    border-top: 1px solid var(--border, #2f2f2f);
    padding-top: 24px;
  }

  .rollouts-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 16px;
  }

  .rollouts-header h2 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-bright);
    margin: 0;
  }

  .rollouts-count {
    font-size: 12px;
    color: var(--muted);
  }

  .rollout-chart {
    margin-bottom: 16px;
  }

  .rollout-chart svg {
    width: 100%;
    height: 140px;
    background: var(--color-c-gray-08, #1c1c1c);
    border-radius: 6px;
  }

  .rollout-chart-meta {
    display: flex;
    gap: 16px;
    margin-top: 6px;
    font-size: 11px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .rollout-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }

  .rollout-table th {
    text-align: left;
    color: var(--muted);
    font-weight: 500;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 6px 10px;
    border-bottom: 1px solid var(--border, #2f2f2f);
  }

  .rollout-table tbody tr {
    cursor: pointer;
  }

  .rollout-table tbody tr:hover {
    background: var(--color-c-gray-10, #2f2f2f);
  }

  .rollout-table tbody tr.expanded {
    background: var(--color-c-gray-10, #2f2f2f);
  }

  .rollout-table td {
    padding: 8px 10px;
    border-bottom: 1px solid var(--border, #2f2f2f);
    font-variant-numeric: tabular-nums;
  }

  .rollout-mean {
    color: var(--text-bright);
  }

  .rollout-detail-row td {
    padding: 12px 10px;
    background: var(--color-c-gray-08, #1c1c1c);
    cursor: default;
  }

  .rollout-sample {
    border-left: 2px solid var(--accent);
    padding: 8px 12px;
    margin-bottom: 12px;
    background: var(--color-c-gray-10, #2f2f2f);
    border-radius: 0 4px 4px 0;
  }

  .rollout-sample-header {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 6px;
  }

  .rollout-sample-score {
    color: var(--text-bright);
    font-variant-numeric: tabular-nums;
  }

  .rollout-sample-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    margin-top: 8px;
    margin-bottom: 2px;
  }

  .rollout-sample-text {
    margin: 0;
    padding: 8px;
    background: var(--color-c-gray-08, #1c1c1c);
    border-radius: 4px;
    font-size: 12px;
    color: var(--text);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 240px;
    overflow: auto;
  }

  .empty {
    color: var(--muted);
    font-size: 13px;
    padding: 16px 0;
  }

  .logs {
    border-top: 1px solid var(--border, #2f2f2f);
    padding-top: 24px;
    margin-top: 24px;
  }

  .logs-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 12px;
  }

  .logs-header h2 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-bright);
    margin: 0;
  }

  .logs-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 9999px;
    background: var(--muted);
  }

  .dot-live {
    background: #4ade80;
    box-shadow: 0 0 0 2px rgba(74, 222, 128, 0.18);
  }

  .dot-warn {
    background: #fbbf24;
  }

  .dot-err {
    background: #f87171;
  }

  .dot-dim {
    background: #6b7280;
  }

  .log-tail {
    background: var(--color-c-gray-08, #0e0e0e);
    border-radius: 6px;
    padding: 8px 12px;
    max-height: 420px;
    overflow-y: auto;
    overflow-x: auto;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
    line-height: 1.45;
    color: var(--text);
  }

  .log-row {
    display: flex;
    gap: 10px;
    white-space: pre;
  }

  .log-task {
    flex-shrink: 0;
    color: var(--muted);
    font-size: 10px;
    min-width: 64px;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .log-line {
    flex: 1;
    white-space: pre-wrap;
    word-break: break-all;
  }

  .log-meta {
    margin-top: 6px;
    font-size: 11px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
    display: flex;
    gap: 6px;
  }

  .log-meta-drop {
    color: #fbbf24;
  }

  .logs-controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
  }

  .log-button {
    background: var(--color-c-gray-10, #2f2f2f);
    color: var(--text);
    border: 1px solid var(--border, #3a3a3a);
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 12px;
    cursor: pointer;
  }

  .log-button:hover:not(:disabled) {
    background: var(--color-c-gray-12, #3a3a3a);
  }

  .log-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .log-search {
    flex: 1;
    min-width: 160px;
    background: var(--color-c-gray-08, #1c1c1c);
    color: var(--text);
    border: 1px solid var(--border, #3a3a3a);
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
    font-family: inherit;
  }

  .log-rate,
  .log-follow {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--muted);
    font-size: 11px;
  }

  .log-rate select {
    background: var(--color-c-gray-08, #1c1c1c);
    color: var(--text);
    border: 1px solid var(--border, #3a3a3a);
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 12px;
  }
</style>
