<script>
  import { CheckCircle2, CircleX, Loader2, OctagonX, MinusCircle } from "lucide-svelte";

  let { status, iconOnly = false, label = null } = $props();

  const STATUS_MAP = {
    completed: "Completed",
    ready: "Ready",
    pending: "Pending",
    running: "Pending",
    stopped: "Stopped",
    cancelled: "Cancelled",
    failed: "Failed",
    inactive: "Inactive",
  };

  let normalizedStatus = $derived.by(() => {
    const s = String(status || "").toLowerCase();
    if (s === "running") return "pending";
    return s in STATUS_MAP ? s : "pending";
  });

  let statusLabel = $derived(label ?? (STATUS_MAP[normalizedStatus] ?? "Pending"));
</script>

<div
  class="status-pill"
  class:icon-only={iconOnly}
  class:status-completed={normalizedStatus === "completed"}
  class:status-ready={normalizedStatus === "ready"}
  class:status-running={normalizedStatus === "running"}
  class:status-pending={normalizedStatus === "pending"}
  class:status-stopped={normalizedStatus === "stopped"}
  class:status-cancelled={normalizedStatus === "cancelled"}
  class:status-failed={normalizedStatus === "failed"}
  class:status-inactive={normalizedStatus === "inactive"}
  aria-label={statusLabel}
>
  {#if normalizedStatus === "completed" || normalizedStatus === "ready"}
    <CheckCircle2 size={14} />
  {:else if normalizedStatus === "stopped" || normalizedStatus === "cancelled"}
    <OctagonX size={14} />
  {:else if normalizedStatus === "failed" || normalizedStatus === "inactive"}
    <CircleX size={14} />
  {:else}
    <span class="live-spinner">
      <Loader2 size={16} />
    </span>
  {/if}
  {#if !iconOnly}
    <span class="status-text">{statusLabel}</span>
  {/if}
</div>

<style>
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
    max-width: 100%;
    white-space: nowrap;
    border-radius: 9999px;
    padding: 3px 10px;
    font-size: 12px;
    line-height: 16px;
    border: 1px solid transparent;
    box-sizing: border-box;
    width: fit-content;
  }

  .status-pill.icon-only {
    min-width: 24px;
    width: 24px;
    padding-inline: 0;
    gap: 0;
    justify-content: center;
  }

  .status-running {
    background: #2f2436;
    color: #d176bd;
    border-color: #3b2a37;
  }

  .status-pending {
    background: #2f2436;
    color: #d176bd;
    border-color: #3b2a37;
  }

  .status-completed,
  .status-ready {
    background: #273823;
    color: #6ac355;
    border-color: #2d4327;
  }

  .status-stopped,
  .status-cancelled {
    background: #3b2f20;
    color: #ffab5e;
    border-color: #5d442d;
  }

  .status-failed,
  .status-inactive {
    background: #3b2020;
    color: #f87171;
    border-color: #5b3333;
  }

  /* Icons carry the status meaning, so they must never be shrunk or clipped
     when the pill is squeezed into a narrow column — only the label truncates. */
  .status-pill :global(svg) {
    flex: 0 0 auto;
  }

  .status-text {
    font-weight: 500;
    font-variant-numeric: tabular-nums;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .live-spinner {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    flex: 0 0 16px;
    animation: status-pill-spin 1s linear infinite;
  }

  .live-spinner :global(svg) {
    display: block;
  }

  @keyframes status-pill-spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }
</style>
