<script>
  let { progress = null, progressLabel = "", stageLabel = "", compact = false } = $props();

  function progressPercent(value) {
    if (!value?.total) return 0;
    const current = Number(value.current);
    const total = Number(value.total);
    if (!Number.isFinite(current) || !Number.isFinite(total) || total <= 0) {
      return 0;
    }
    return Math.max(0, Math.min(100, (current / total) * 100));
  }
</script>

<div class="framework-stage-progress" class:compact>
  {#if progress}
    <span class="progress-track" aria-hidden="true">
      <span
        class="progress-fill"
        style={`width: ${progressPercent(progress)}%`}
      ></span>
    </span>
  {/if}
  {#if progressLabel}
    <span class="progress-label">{progressLabel}</span>
  {/if}
  {#if stageLabel}
    <span class="stage-label">{stageLabel}</span>
  {/if}
</div>

<style>
  .framework-stage-progress {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
    min-width: 0;
    width: 100%;
  }

  .framework-stage-progress.compact {
    gap: 3px;
  }

  .progress-track {
    display: block;
    width: 100%;
    height: 6px;
    background: var(--color-c-gray-10, #2f2f2f);
    border-radius: 9999px;
    overflow: hidden;
  }

  .framework-stage-progress.compact .progress-track {
    height: 4px;
  }

  .progress-fill {
    display: block;
    height: 100%;
    min-width: 2px;
    border-radius: inherit;
    background: var(--accent);
  }

  .progress-label {
    color: var(--muted);
    font-size: 11px;
    font-weight: 500;
    line-height: 12px;
    font-variant-numeric: tabular-nums;
    min-width: 0;
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .stage-label {
    color: color-mix(in srgb, var(--accent) 42%, var(--muted) 58%);
    font-size: 11px;
    font-weight: 500;
    line-height: 12px;
    font-variant-numeric: tabular-nums;
    letter-spacing: 0;
    text-shadow: 0 0 10px color-mix(in srgb, var(--accent) 18%, transparent);
    animation: stage-label-flash 1.2s ease-in-out infinite;
    min-width: 0;
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  @keyframes stage-label-flash {
    0%,
    100% {
      opacity: 0.48;
      filter: brightness(0.95);
    }

    50% {
      opacity: 1;
      filter: brightness(1.12);
    }
  }
</style>
