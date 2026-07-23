<script>
  import { Download, ZoomIn, ZoomOut } from "lucide-svelte";
  import { phaseBreakdown } from "../lib/asyncTimingProfile.js";

  let {
    stepTimes = null,
    substepTimes = null,
    layout = "rows",
    asyncMode = false,
    downloadName = "step_substep_times.json",
  } = $props();

  const SUBSTEP_LABELS = {
    evaluate_rollouts: "Eval (before)",
    generate_rollouts: "Generate rollouts",
    custom_reward_function: "Custom reward function",
    offload_rollout: "Offload rollout",
    compute_log_probs: "Compute log probs",
    train_model: "Train model",
    optimizer_step: "Optimizer step",
    checkpoint_save: "Checkpoint save",
    offload_train: "Offload train",
    weight_sync: "Weight sync",
    evaluate_rollouts_end: "Eval (after)",
  };

  const SUBSTEP_COLORS = {
    evaluate_rollouts: "#60a5fa",
    generate_rollouts: "#34d399",
    custom_reward_function: "#2dd4bf",
    offload_rollout: "#a78bfa",
    compute_log_probs: "#fbbf24",
    train_model: "#fb923c",
    optimizer_step: "#f87171",
    weight_sync: "#22d3ee",
    checkpoint_save: "#f472b6",
    offload_train: "#c084fc",
    evaluate_rollouts_end: "#818cf8",
  };

  const TRAINING_SUBSTEP_LABELS = {
    data_preprocess: "Load & transfer training batch",
    compute_log_probs: "Policy log probabilities",
    reference_log_probs: "Reference log probabilities",
    teacher_log_probs: "Teacher log probabilities",
    value_inference: "Critic value inference",
    forward_backward: "Forward / backward",
    optimizer_step: "Optimizer step",
    reference_model_update: "Update reference model",
    training_model_wake: "Load training model",
    training_model_offload: "Offload training model",
  };

  const ASYNC_SUBSTEP_LABELS = {
    ...SUBSTEP_LABELS,
    generate_rollouts: "Rollout generation",
    custom_reward: "Custom reward function",
    training: "Training",
    ...TRAINING_SUBSTEP_LABELS,
  };

  const TRAINING_SUBSTEP_COLORS = {
    data_preprocess: "var(--color-c-dataviz-paired-3, #d6a84b)",
    compute_log_probs: SUBSTEP_COLORS.compute_log_probs,
    reference_log_probs: "#fb923c",
    teacher_log_probs: "#c084fc",
    value_inference: "var(--color-c-dataviz-paired-1, #78a967)",
    forward_backward: "var(--color-c-dataviz-paired-4, #6cabc1)",
    optimizer_step: "var(--color-c-dataviz-paired-7, #8956fa)",
    reference_model_update: "var(--color-c-dataviz-paired-6, #b48ad6)",
    training_model_wake: SUBSTEP_COLORS.offload_rollout,
    training_model_offload: SUBSTEP_COLORS.offload_train,
  };

  const ASYNC_SUBSTEP_COLORS = {
    ...SUBSTEP_COLORS,
    custom_reward: "#f59e0b",
    training: "var(--color-c-dataviz-primary-7, #648fe0)",
    ...TRAINING_SUBSTEP_COLORS,
  };

  const TRAINING_SUBSTEP_DESCRIPTIONS = {
    data_preprocess:
      "Fetches rollout data, shards it for the training workers, builds training tensors, and enqueues their GPU transfers.",
    compute_log_probs:
      "A current-policy forward pass used when Slime cannot reuse log probabilities from the training loss.",
    reference_log_probs:
      "A frozen reference-policy forward pass used to calculate KL-related log probabilities.",
    teacher_log_probs:
      "A teacher-model forward pass used for on-policy distillation.",
    value_inference:
      "A critic forward pass that computes value estimates used by the value loss and actor advantages.",
    forward_backward:
      "Host interval through forward/backward and gradient preparation, ending when optimizer.step() is called.",
    optimizer_step:
      "Host interval of optimizer.step(). CUDA is asynchronous, so it can include waiting for earlier GPU work and is not isolated optimizer-kernel time.",
    reference_model_update:
      "Copies the current actor weights into the reference-model backup when the configured reference update interval fires.",
    training_model_wake:
      "Restores an offloaded training model and its distributed process groups before this worker trains.",
    training_model_offload:
      "Releases training-model memory and distributed process groups after this worker finishes training.",
  };

  const ASYNC_SUBSTEP_DESCRIPTIONS = {
    custom_reward:
      "Execution time inside the run-level custom reward callable for indexed samples or groups. Concurrent calls are shown as wall-clock coverage.",
    training:
      "Wall time waiting for the training workers, from dispatch through their return.",
    ...TRAINING_SUBSTEP_DESCRIPTIONS,
    checkpoint_save: "Checkpoint persistence, shown separately from training.",
  };

  const SUBSTEP_ORDER = Object.keys(SUBSTEP_LABELS);
  const ASYNC_SUBSTEP_ORDER = [
    "evaluate_rollouts",
    "generate_rollouts",
    "custom_reward",
    "training",
    "data_preprocess",
    "compute_log_probs",
    "reference_log_probs",
    "teacher_log_probs",
    "value_inference",
    "forward_backward",
    "optimizer_step",
    "reference_model_update",
    "training_model_wake",
    "training_model_offload",
    "checkpoint_save",
    "weight_sync",
    "offload_rollout",
    "offload_train",
    "evaluate_rollouts_end",
  ];
  const FALLBACK_COLORS = [
    "#60a5fa",
    "#34d399",
    "#a78bfa",
    "#fbbf24",
    "#f472b6",
    "#22d3ee",
  ];

  // Timeline zoom bounds: 1 = fit-to-width, MAX_ZOOM = deepest magnification.
  const MIN_ZOOM = 1;
  const MAX_ZOOM = 64;
  const ZOOM_BTN_FACTOR = 1.5;
  const WHEEL_SENSITIVITY = 0.0015;

  function labelFor(name) {
    return SUBSTEP_LABELS[name] || name.replace(/_/g, " ");
  }

  function colorFor(name) {
    return SUBSTEP_COLORS[name] || "var(--color-c-gray-40, #5e5e5e)";
  }

  function asyncFallbackColor(name) {
    let hash = 0;
    for (const character of name) hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
    return FALLBACK_COLORS[hash % FALLBACK_COLORS.length];
  }

  function asyncLabelFor(phase) {
    const name = typeof phase === "string" ? phase : phase.name;
    return phase?.displayName || ASYNC_SUBSTEP_LABELS[name] || labelFor(name);
  }

  function asyncColorFor(phase) {
    const name = typeof phase === "string" ? phase : phase.name;
    return ASYNC_SUBSTEP_COLORS[name] || asyncFallbackColor(name);
  }

  // Durations are float seconds; keep up to 3 decimals (trailing zeros trimmed).
  function fmtSecs(s) {
    if (s == null) return "—";
    const n = Number(s);
    if (!Number.isFinite(n)) return "—";
    const trim = (x) => x.toFixed(3).replace(/\.?0+$/, "");
    if (n >= 60) {
      const m = Math.floor(n / 60);
      return `${m}m ${trim(n - m * 60)}s`;
    }
    return `${trim(n)}s`;
  }

  function downloadJson() {
    const payload = {
      step_times: stepTimes || {},
      substep_times: substepTimes || {},
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = downloadName;
    a.click();
    URL.revokeObjectURL(url);
  }

  let steps = $derived.by(() => {
    const stepKeys = Object.keys(stepTimes || {});
    const subKeys = Object.keys(substepTimes || {});
    const keys = Array.from(new Set([...stepKeys, ...subKeys]));
    const out = keys.map((k) => {
      const st = (stepTimes || {})[k] || null;
      const subs = (substepTimes || {})[k] || {};
      const substeps = Object.entries(subs)
        .map(([name, v]) => {
          const storedIntervals = v?.intervals;
          const hasIntervals =
            Array.isArray(storedIntervals) && storedIntervals.length > 0;
          const values = hasIntervals ? storedIntervals : [v];
          const descriptor = hasIntervals ? storedIntervals[0] : v;
          const segments = [];
          for (const [intervalIndex, value] of values.entries()) {
            if (value?.start == null || value?.duration_s == null) continue;
            const start = Number(value?.start);
            const duration = Number(value?.duration_s);
            if (!Number.isFinite(start) || !Number.isFinite(duration) || duration < 0) {
              continue;
            }
            const occurrenceId = value.step_id;
            const role = value.training_role;
            const rank = value.slowest_rank ?? value.training_rank;
            const worldSize = value.training_world_size;
            segments.push({
              key: `${name}-${role ?? ""}-${rank ?? ""}-${occurrenceId ?? ""}-${intervalIndex}`,
              occurrenceId:
                hasIntervals &&
                Number.isInteger(occurrenceId) &&
                occurrenceId >= 0
                  ? occurrenceId
                  : null,
              role:
                hasIntervals && typeof role === "string"
                  ? role
                  : null,
              slowestRank:
                hasIntervals && Number.isInteger(value.slowest_rank)
                  ? value.slowest_rank
                  : null,
              trainingRank:
                hasIntervals && Number.isInteger(value.training_rank)
                  ? value.training_rank
                  : null,
              reportedRankCount:
                hasIntervals && Number.isInteger(value.reported_rank_count)
                  ? value.reported_rank_count
                  : null,
              worldSize:
                hasIntervals && Number.isInteger(worldSize)
                  ? worldSize
                  : null,
              timelineLane:
                typeof value?.timeline_lane === "string"
                  ? value.timeline_lane
                  : (descriptor?.timeline_lane ?? null),
              parentPhase:
                typeof value?.parent_phase === "string"
                  ? value.parent_phase
                  : (descriptor?.parent_phase ?? null),
              displayName:
                typeof value?.display_name === "string"
                  ? value.display_name
                  : descriptor?.display_name,
              start,
              duration,
              end: start + duration,
            });
          }
          return {
            name,
            start: v?.start ?? null,
            duration: v?.duration_s ?? null,
            timelineLane:
              typeof descriptor?.timeline_lane === "string"
                ? descriptor.timeline_lane
                : null,
            parentPhase:
              typeof descriptor?.parent_phase === "string"
                ? descriptor.parent_phase
                : null,
            displayName:
              typeof descriptor?.display_name === "string" ? descriptor.display_name : null,
            hasIntervals,
            segments,
          };
        })
        .sort((a, b) => (a.start ?? 0) - (b.start ?? 0));
      const segmentCounts = new Map();
      for (const sub of substeps) {
        for (const item of sub.segments) {
          const countKey = `${sub.name}:${item.role ?? ""}`;
          segmentCounts.set(countKey, (segmentCounts.get(countKey) || 0) + 1);
        }
      }
      for (const sub of substeps) {
        for (const item of sub.segments) {
          item.repeated = segmentCounts.get(`${sub.name}:${item.role ?? ""}`) > 1;
        }
      }
      const syncSubsteps = [];
      const details = [];
      for (const sub of substeps) {
        if (sub.hasIntervals) {
          for (const item of sub.segments) {
            if (item.parentPhase != null) continue;
            syncSubsteps.push({
              name: sub.name,
              ...item,
            });
          }
          const detail = sub.segments.find((item) => item.parentPhase != null);
          if (detail) {
            details.push({
              key: sub.name,
              name: sub.name,
              label: detail.displayName || labelFor(sub.name),
              duration: sub.duration,
            });
          }
        } else {
          syncSubsteps.push({
            key: sub.name,
            name: sub.name,
            start: sub.start,
            duration: sub.duration,
            role: null,
            occurrenceId: null,
            displayName: sub.displayName,
            repeated: false,
          });
        }
      }
      syncSubsteps.sort((a, b) => (a.start ?? 0) - (b.start ?? 0));
      const step = {
        key: k,
        n: Number.isFinite(Number(k)) ? Number(k) : k,
        rolloutId: Number.isFinite(Number(k)) ? Math.max(0, Number(k) - 1) : k,
        duration: st?.duration_s ?? null,
        substeps,
        syncSubsteps,
        details,
      };
      step.timeline = [];
      let segmentIndex = 0;
      for (const sub of substeps) {
        for (const item of sub.segments) {
          step.timeline.push({
            step,
            sub: {
              name: sub.name,
              ...item,
              segmentId: `${step.key}:${sub.name}:${segmentIndex++}`,
            },
            start: item.start,
            end: item.end,
          });
        }
      }
      step.timeline.sort((a, b) => a.start - b.start);
      return step;
    });
    out.sort((a, b) => (Number(a.key) || 0) - (Number(b.key) || 0));
    return out;
  });

  let hasData = $derived(steps.length > 0);
  let phaseDefinitions = $derived.by(() => {
    const definitions = {};
    for (const step of steps) {
      for (const sub of step.substeps) definitions[sub.name] ??= sub;
    }
    return definitions;
  });

  function isTrainingChild(sub) {
    return sub.parentPhase === "training" && sub.timelineLane === "training";
  }

  function isNestedDetail(step, sub) {
    return (
      sub.parentPhase != null &&
      sub.parentPhase !== "training" &&
      step.substeps.some((phase) => phase.name === sub.parentPhase)
    );
  }

  function phaseProfile(step, parent) {
    const breakdown = phaseBreakdown(step.timeline, parent);
    if (!breakdown) return null;
    const rows = breakdown.phases.map((phase) => {
      return {
        key: phase.key,
        label: tooltipLabel(step, {
          ...phase.phase,
          occurrenceId: null,
          repeated: false,
        }),
        duration: phase.duration,
      };
    });
    const largest = rows.reduce((current, row) =>
      row.duration > current.duration ? row : current,
    );
    const summary =
      largest.duration >= breakdown.total / 2
        ? `Mostly ${largest.label.toLowerCase()} · ${fmtSecs(largest.duration)} of ${fmtSecs(breakdown.total)}`
        : `${rows.length} measured ${rows.length === 1 ? "phase" : "phases"} · ${fmtSecs(breakdown.measured)} of ${fmtSecs(breakdown.total)}`;
    if (breakdown.other > 0.0005) {
      rows.push({ key: "other", label: "Other work", duration: breakdown.other });
    }
    return { summary, rows };
  }

  function tooltipLabel(_step, sub) {
    const label = asyncMode
      ? asyncLabelFor(sub)
      : sub.displayName || labelFor(sub.name);
    const roleName =
      sub.role && sub.role !== "driver"
        ? `${sub.role[0].toUpperCase()}${sub.role.slice(1)}`
        : "";
    const role =
      roleName && !label.toLowerCase().startsWith(`${roleName.toLowerCase()} `)
        ? `${roleName} `
        : "";
    const update =
      sub.occurrenceId != null && sub.repeated
        ? ` ${sub.occurrenceId + 1}`
        : "";
    return `${role}${label}${update}`;
  }

  function tooltipDescription(sub) {
    return asyncMode ? ASYNC_SUBSTEP_DESCRIPTIONS[sub.name] : null;
  }

  function rankSummary(sub) {
    if (sub.slowestRank == null || sub.reportedRankCount == null) {
      if (sub.trainingRank == null) return null;
      return sub.worldSize == null
        ? `Training rank ${sub.trainingRank}`
        : `Training rank ${sub.trainingRank} of ${sub.worldSize}`;
    }
    if (sub.worldSize == null) {
      return `Slowest of ${sub.reportedRankCount} reported ranks · global rank ${sub.slowestRank}`;
    }
    if (sub.reportedRankCount < sub.worldSize) {
      return `Slowest reported: global rank ${sub.slowestRank} · ${sub.reportedRankCount}/${sub.worldSize} ranks reported`;
    }
    return `Slowest of ${sub.worldSize} ranks · global rank ${sub.slowestRank}`;
  }

  function trainingChildIntervals(step) {
    return step.timeline
      .filter((interval) => isTrainingChild(interval.sub))
      .sort((a, b) => a.start - b.start);
  }

  function topLevelSubsteps(step) {
    const hasUpdates = trainingChildIntervals(step).length > 0;
    return hasUpdates
      ? step.substeps.filter(
          (sub) =>
            sub.name !== "training" &&
            !isTrainingChild(sub) &&
            !isNestedDetail(step, sub),
        )
      : step.substeps.filter((sub) => !isNestedDetail(step, sub));
  }

  let asyncTimeline = $derived.by(() => {
    const rollout = [];
    const reward = [];
    const training = [];
    const trainingWindows = [];
    const coordination = [];
    for (const step of steps) {
      const hasUpdates = trainingChildIntervals(step).length > 0;
      for (const segment of step.timeline) {
        if (isTrainingChild(segment.sub)) training.push(segment);
        else if (segment.sub.name === "training" && hasUpdates) {
          trainingWindows.push(segment);
        } else if (segment.sub.timelineLane === "rollout") rollout.push(segment);
        else if (segment.sub.timelineLane === "reward") reward.push(segment);
        else if (segment.sub.timelineLane === "training") training.push(segment);
        else if (segment.sub.timelineLane === "coordination") coordination.push(segment);
        else if (segment.sub.name === "training") training.push(segment);
        else coordination.push(segment);
      }
    }
    let start = Infinity;
    let end = -Infinity;
    for (const segments of [rollout, reward, trainingWindows, training, coordination]) {
      for (const segment of segments) {
        start = Math.min(start, segment.start);
        end = Math.max(end, segment.end);
      }
    }
    const trainingRoles = [
      ...new Set(training.map((segment) => segment.sub.role).filter(Boolean)),
    ].sort((a, b) => {
      const order = { actor: 0, critic: 1 };
      return (order[a] ?? 2) - (order[b] ?? 2) || a.localeCompare(b);
    });
    const splitTrainingRoles = trainingRoles.length > 1;
    const trainingByRole = splitTrainingRoles
      ? trainingRoles.map((role) => ({
          role,
          segments: training.filter((segment) => segment.sub.role === role),
        }))
      : [];
    const visibleTraining = splitTrainingRoles
      ? training.filter((segment) => !segment.sub.role)
      : training;
    if (!Number.isFinite(start) || !Number.isFinite(end)) {
      return {
        start: 0,
        duration: 1,
        rollout,
        reward,
        visibleTraining,
        trainingByRole,
        trainingWindows,
        coordination,
      };
    }
    return {
      start,
      duration: Math.max(end - start, 0.001),
      rollout,
      reward,
      visibleTraining,
      trainingByRole,
      trainingWindows,
      coordination,
    };
  });

  let legend = $derived.by(() => {
    const seen = new Set();
    if (asyncMode) {
      for (const step of steps) {
        for (const sub of topLevelSubsteps(step)) seen.add(sub.name);
        for (const segment of step.timeline) {
          if (segment.sub.parentPhase != null) {
            seen.add(segment.sub.name);
          }
        }
      }
      if (asyncTimeline.trainingWindows.length) seen.add("training");
    } else {
      for (const step of steps) {
        for (const sub of step.syncSubsteps) {
          if (SUBSTEP_ORDER.includes(sub.name)) seen.add(sub.name);
        }
      }
    }
    const order = asyncMode ? ASYNC_SUBSTEP_ORDER : SUBSTEP_ORDER;
    const known = order.filter((name) => seen.has(name));
    if (!asyncMode) return known;
    return [...known, ...[...seen].filter((name) => !order.includes(name)).sort()];
  });

  let tip = $state(null);
  let pinned = $state(false);

  // ── Timeline zoom / pan state ────────────────────────────────────────
  let zoom = $state(1);
  let viewport = $state(null);

  function stepWeight(step) {
    const subTotal = step.syncSubsteps.reduce((acc, s) => acc + (s.duration ?? 0), 0);
    if (subTotal > 0) return subTotal;
    if (step.duration != null && step.duration > 0) return step.duration;
    return 1;
  }

  function setZoom(next, anchorX = null) {
    const clamped = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, next));
    if (clamped === zoom) return;
    if (viewport) {
      const rect = viewport.getBoundingClientRect();
      const cursorX = anchorX == null ? rect.width / 2 : anchorX - rect.left;
      const contentX = viewport.scrollLeft + cursorX;
      const scale = clamped / zoom;
      zoom = clamped;
      requestAnimationFrame(() => {
        viewport.scrollLeft = contentX * scale - cursorX;
      });
    } else {
      zoom = clamped;
    }
  }

  function handleWheel(e) {
    // Let horizontal trackpad gestures pan natively; vertical wheel zooms.
    if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
    e.preventDefault();
    setZoom(zoom * Math.exp(-e.deltaY * WHEEL_SENSITIVITY), e.clientX);
  }

  // Wheel listeners are passive by default; zooming needs preventDefault.
  function wheelZoom(node) {
    node.addEventListener("wheel", handleWheel, { passive: false });
    return {
      destroy() {
        node.removeEventListener("wheel", handleWheel);
      },
    };
  }

  function isActive(step, sub) {
    if (tip?.segmentId != null || sub.segmentId != null) {
      return tip?.segmentId === sub.segmentId;
    }
    if (tip?.key != null || sub.key != null) {
      return tip?.rolloutId === step.rolloutId && tip?.key === sub.key;
    }
    return (
      tip &&
      tip.rolloutId === step.rolloutId &&
      tip.name === sub.name &&
      tip.role === (sub.role ?? null) &&
      tip.occurrenceId === (sub.occurrenceId ?? null)
    );
  }

  function tooltipFor(e, step, sub) {
    const bounds = e.currentTarget?.getBoundingClientRect();
    const hasPointerCoordinates =
      e.type !== "keydown" &&
      typeof e.clientX === "number" &&
      typeof e.clientY === "number";
    return {
      x: hasPointerCoordinates ? e.clientX : (bounds?.left ?? 0) + (bounds?.width ?? 0) / 2,
      y: hasPointerCoordinates ? e.clientY : (bounds?.top ?? 0),
      rolloutId: step.rolloutId,
      role: sub.role ?? null,
      occurrenceId: sub.occurrenceId ?? null,
      segmentId: sub.segmentId ?? null,
      key: sub.key ?? null,
      name: sub.name,
      label: tooltipLabel(step, sub),
      stepLabel: asyncMode ? `Rollout ${step.rolloutId}` : `Step ${step.n}`,
      dur: sub.duration,
      rankSummary: rankSummary(sub),
      description: tooltipDescription(sub),
      profile: phaseProfile(step, sub),
    };
  }

  function showTip(e, step, sub) {
    if (pinned) return;
    tip = tooltipFor(e, step, sub);
  }

  function moveTip(e) {
    if (pinned || !tip) return;
    tip = { ...tip, x: e.clientX, y: e.clientY };
  }

  function hideTip() {
    if (pinned) return;
    tip = null;
  }

  function pinTip(e, step, sub) {
    e.stopPropagation();
    if (pinned && isActive(step, sub)) {
      pinned = false;
      tip = null;
      return;
    }
    pinned = true;
    tip = tooltipFor(e, step, sub);
  }

  function clearPin() {
    if (!pinned) return;
    pinned = false;
    tip = null;
  }
</script>

<svelte:window onclick={clearPin} />

{#snippet segment(step, sub)}
  <div
    class="seg"
    class:seg-null={sub.duration == null}
    class:active={pinned && isActive(step, sub)}
    style:flex-grow={sub.duration == null ? undefined : Math.max(sub.duration, 0.01)}
    style:background={sub.duration == null ? undefined : colorFor(sub.name)}
    role="button"
    tabindex="0"
    aria-label={`${tooltipLabel(step, sub)}, ${fmtSecs(sub.duration)}`}
    onmouseenter={(e) => showTip(e, step, sub)}
    onmousemove={moveTip}
    onmouseleave={hideTip}
    onclick={(e) => pinTip(e, step, sub)}
    onkeydown={(e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        pinTip(e, step, sub);
      }
    }}
  ></div>
{/snippet}

{#snippet timingDetails(step)}
  {#if step.details.length}
    <div class="timing-details">
      {#each step.details as detail (detail.key)}
        <div class="timing-detail">
          <span class="swatch" style:background={colorFor(detail.name)}></span>
          <span class="timing-detail-name">{detail.label}</span>
          <span class="timing-detail-duration">{fmtSecs(detail.duration)}</span>
        </div>
      {/each}
    </div>
  {/if}
{/snippet}

{#snippet asyncSegment(segment)}
  {@const trainingChild = isTrainingChild(segment.sub)}
  {@const nestedDetail = segment.sub.parentPhase != null && segment.sub.parentPhase !== "training"}
  <div
    class="seg async-seg"
    class:training-inner-seg={trainingChild}
    class:nested-detail-seg={nestedDetail}
    class:active={pinned && isActive(segment.step, segment.sub)}
    style={`left:${((segment.start - asyncTimeline.start) / asyncTimeline.duration) * 100}%;width:${(segment.sub.duration / asyncTimeline.duration) * 100}%;background:${asyncColorFor(segment.sub)}`}
    role="button"
    tabindex="0"
    aria-label={`${tooltipLabel(segment.step, segment.sub)}, ${fmtSecs(segment.sub.duration)}`}
    onmouseenter={(e) => showTip(e, segment.step, segment.sub)}
    onmousemove={moveTip}
    onmouseleave={hideTip}
    onclick={(e) => pinTip(e, segment.step, segment.sub)}
    onkeydown={(e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        pinTip(e, segment.step, segment.sub);
      }
    }}
  ></div>
{/snippet}

{#snippet trainingWindow(segment)}
  <div
    class="training-window"
    class:active={pinned && isActive(segment.step, segment.sub)}
    style={`left:${((segment.start - asyncTimeline.start) / asyncTimeline.duration) * 100}%;width:${(segment.sub.duration / asyncTimeline.duration) * 100}%;--training-color:${asyncColorFor(segment.sub)}`}
    role="button"
    tabindex="0"
    aria-label={`${tooltipLabel(segment.step, segment.sub)}, ${fmtSecs(segment.sub.duration)}`}
    onmouseenter={(e) => showTip(e, segment.step, segment.sub)}
    onmousemove={moveTip}
    onmouseleave={hideTip}
    onclick={(e) => pinTip(e, segment.step, segment.sub)}
    onkeydown={(e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        pinTip(e, segment.step, segment.sub);
      }
    }}
  ></div>
{/snippet}

{#snippet asyncLane(label, segments, windows = [])}
  {#if segments.length || windows.length}
    <div class="async-lane">
      <div class="async-lane-label tl-step-name">{label}</div>
      <div class="bar tl-bar async-lane-track">
        {#each windows as segment (segment.sub.segmentId)}
          {@render trainingWindow(segment)}
        {/each}
        {#each segments as segment (segment.sub.segmentId)}
          {@render asyncSegment(segment)}
        {/each}
      </div>
    </div>
  {/if}
{/snippet}

{#if hasData}
  <div class="step-timings">
    {#if legend.length || layout === "timeline"}
      <div class="legend-row">
        <div class="legend">
          {#each legend as name (name)}
            <span class="legend-item">
              <span class="swatch" style:background={asyncMode ? asyncColorFor(phaseDefinitions[name] || name) : colorFor(name)}></span>
              {asyncMode ? asyncLabelFor(phaseDefinitions[name] || name) : labelFor(name)}
            </span>
          {/each}
        </div>
        {#if layout === "timeline"}
          <div class="tl-toolbar">
            <div class="zoom-controls">
              <button
                class="zoom-btn"
                onclick={() => setZoom(zoom / ZOOM_BTN_FACTOR)}
                disabled={zoom <= MIN_ZOOM}
                title="Zoom out"
              >
                <ZoomOut size={13} />
              </button>
              <button
                class="zoom-level"
                onclick={() => setZoom(MIN_ZOOM)}
                disabled={zoom <= MIN_ZOOM}
                title="Reset zoom to fit"
              >
                {zoom >= 10 ? Math.round(zoom) : zoom.toFixed(1).replace(/\.0$/, "")}×
              </button>
              <button
                class="zoom-btn"
                onclick={() => setZoom(zoom * ZOOM_BTN_FACTOR)}
                disabled={zoom >= MAX_ZOOM}
                title="Zoom in"
              >
                <ZoomIn size={13} />
              </button>
            </div>
            <button
              class="dl-btn"
              onclick={downloadJson}
              title="Download step + substep times as JSON"
            >
              <Download size={13} />
              Download JSON
            </button>
          </div>
        {/if}
      </div>
    {/if}

    {#if layout === "timeline" && asyncMode}
      <div class="tl-viewport" bind:this={viewport} use:wheelZoom>
        <div class="async-timeline" style:width={`${zoom * 100}%`}>
          <div class="async-axis">
            <span class="tl-step-name">Async wall-clock timeline</span>
            <span class="tl-step-dur">{fmtSecs(asyncTimeline.duration)}</span>
          </div>
          {@render asyncLane("Rollout", asyncTimeline.rollout)}
          {@render asyncLane("Custom reward", asyncTimeline.reward)}
          {@render asyncLane(
            "Training",
            asyncTimeline.visibleTraining,
            asyncTimeline.trainingWindows,
          )}
          {#each asyncTimeline.trainingByRole as lane (lane.role)}
            {@render asyncLane(
              `${lane.role[0].toUpperCase()}${lane.role.slice(1)} training`,
              lane.segments,
            )}
          {/each}
          {@render asyncLane("Coordination / I/O", asyncTimeline.coordination)}
        </div>
      </div>
      <div class="tl-hint">
        Blue outline = total training wall time · inner bars are reported operations · scroll to zoom
      </div>
    {:else if layout === "timeline"}
      <div class="tl-viewport" bind:this={viewport} use:wheelZoom>
        <div class="tl-track" style:width={`${zoom * 100}%`}>
          {#each steps as step (step.key)}
            <div class="tl-step" style:flex-grow={stepWeight(step)}>
              <div class="tl-step-head">
                <span class="tl-step-name">Step {step.n}</span>
                <span class="tl-step-dur">{fmtSecs(step.duration)}</span>
              </div>
              {#if step.syncSubsteps.length}
                <div class="bar tl-bar">
                  {#each step.syncSubsteps as sub (sub.key)}
                    {@render segment(step, sub)}
                  {/each}
                </div>
              {:else}
                <div class="bar tl-bar bar-empty"></div>
              {/if}
              {@render timingDetails(step)}
            </div>
          {/each}
        </div>
      </div>
      <div class="tl-hint">Scroll to zoom · shift-scroll or drag the scrollbar to pan</div>
    {:else}
      {#each steps as step (step.key)}
        <div class="step-row">
          <div class="step-head">
            <span class="step-name">Step {step.n}</span>
            <span class="step-dur">{fmtSecs(step.duration)}</span>
          </div>
          {#if step.syncSubsteps.length}
            <div class="bar">
              {#each step.syncSubsteps as sub (sub.key)}
                {@render segment(step, sub)}
              {/each}
            </div>
          {:else}
            <div class="bar bar-empty"></div>
          {/if}
          {@render timingDetails(step)}
        </div>
      {/each}
    {/if}
  </div>

  {#if tip}
    <div class="tg-tip" class:pinned style:left={`${tip.x}px`} style:top={`${tip.y}px`}>
      <span class="tg-tip-step">{tip.stepLabel}</span>
      <span class="tg-tip-name">{tip.label}</span>
      <span class="tg-tip-dur">
        {tip.dur == null ? "unknown (report dropped)" : fmtSecs(tip.dur)}
      </span>
      {#if tip.rankSummary}
        <span class="tg-tip-rank">{tip.rankSummary}</span>
      {/if}
      {#if tip.description}
        <span class="tg-tip-description">{tip.description}</span>
      {/if}
      {#if tip.profile}
        <span class="tg-tip-summary">{tip.profile.summary}</span>
        <span class="tg-tip-profile">
          {#each tip.profile.rows as row (row.key)}
            <span>{row.label}</span>
            <span>{fmtSecs(row.duration)}</span>
          {/each}
        </span>
      {/if}
    </div>
  {/if}
{/if}

<style>
  .step-timings {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .async-timeline {
    display: flex;
    flex-direction: column;
    gap: 3px;
    min-width: 100%;
    padding-bottom: 2px;
  }

  .async-axis,
  .async-lane {
    display: grid;
    grid-template-columns: 128px minmax(0, 1fr);
    gap: 8px;
  }

  .async-axis {
    font-size: 10px;
    line-height: 14px;
    margin-bottom: 3px;
  }

  .async-axis span:last-child {
    grid-column: 2;
    justify-self: end;
  }

  .async-lane {
    align-items: center;
  }

  .async-lane-label {
    font-size: 10px;
    line-height: 14px;
    overflow: hidden;
    text-align: right;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .tl-bar.async-lane-track {
    position: relative;
    height: 14px;
  }

  .async-seg {
    position: absolute;
    top: 0;
    z-index: 2;
    min-width: 1px;
  }

  .async-seg.training-inner-seg {
    top: 2px;
    z-index: 3;
    height: calc(100% - 4px);
    border-radius: 1px;
  }

  .async-seg.nested-detail-seg {
    top: 4px;
    z-index: 4;
    height: calc(100% - 8px);
    border-radius: 1px;
  }

  .training-window {
    position: absolute;
    top: 0;
    z-index: 1;
    height: 100%;
    border: 1px solid var(--training-color);
    border-radius: 3px;
    background: color-mix(in srgb, var(--training-color) 28%, transparent);
    cursor: pointer;
  }

  .training-window.active {
    outline: 1px solid var(--text-bright);
    outline-offset: -1px;
  }

  .legend-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 4px;
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 14px;
    font-size: 11px;
    color: var(--muted);
  }

  .dl-btn {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    flex-shrink: 0;
    background: none;
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 4px;
    color: var(--muted);
    font-size: 11px;
    padding: 3px 8px;
    cursor: pointer;
  }

  .dl-btn:hover {
    color: var(--text);
    border-color: var(--border-strong, #4a4a4a);
  }

  .legend-item {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }

  .swatch {
    width: 9px;
    height: 9px;
    border-radius: 2px;
    flex-shrink: 0;
  }

  .step-row {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .step-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
    font-size: 12px;
  }

  .step-name {
    color: var(--text-bright);
    font-weight: 500;
  }

  .step-dur {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .timing-details {
    display: flex;
    flex-wrap: wrap;
    gap: 4px 12px;
  }

  .timing-detail {
    display: flex;
    align-items: center;
    gap: 5px;
    min-width: 0;
    color: var(--muted);
    font-size: 11px;
  }

  .timing-detail-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .timing-detail-duration {
    color: var(--text-bright);
    font-variant-numeric: tabular-nums;
  }

  .bar {
    display: flex;
    height: 14px;
    border-radius: 3px;
    overflow: hidden;
    background: var(--color-c-gray-08, #1c1c1c);
    gap: 1px;
  }

  .bar-empty {
    background: var(--color-c-gray-10, #2f2f2f);
  }

  .seg {
    min-width: 2px;
    height: 100%;
    cursor: pointer;
    transition: filter 0.1s ease;
  }

  .seg:hover {
    filter: brightness(1.25);
  }

  .seg.active {
    outline: 1px solid var(--text-bright, #fff);
    outline-offset: -1px;
    filter: brightness(1.3);
  }

  /* Dropped substep: visible but doesn't distort the proportional widths. */
  .seg-null {
    flex: 0 0 16px;
    background: repeating-linear-gradient(
      45deg,
      var(--color-c-gray-20, #3a3a3a),
      var(--color-c-gray-20, #3a3a3a) 3px,
      var(--color-c-gray-10, #2f2f2f) 3px,
      var(--color-c-gray-10, #2f2f2f) 6px
    );
  }

  /* ── Timeline (full-width zoomable bar across all steps) ─────────────── */
  .tl-toolbar {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
  }

  .zoom-controls {
    display: inline-flex;
    align-items: center;
    border: 1px solid var(--border, #2f2f2f);
    border-radius: 4px;
    overflow: hidden;
  }

  .zoom-btn,
  .zoom-level {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: none;
    border: none;
    color: var(--muted);
    font-size: 11px;
    padding: 3px 7px;
    cursor: pointer;
  }

  .zoom-level {
    min-width: 38px;
    border-left: 1px solid var(--border, #2f2f2f);
    border-right: 1px solid var(--border, #2f2f2f);
    font-variant-numeric: tabular-nums;
  }

  .zoom-btn:hover:not(:disabled),
  .zoom-level:hover:not(:disabled) {
    color: var(--text);
    background: var(--color-c-gray-08, #1c1c1c);
  }

  .zoom-btn:disabled,
  .zoom-level:disabled {
    opacity: 0.4;
    cursor: default;
  }

  .tl-viewport {
    overflow-x: auto;
    overflow-y: hidden;
    padding-bottom: 6px;
    overscroll-behavior-x: contain;
  }

  .tl-track {
    display: flex;
    gap: 3px;
    min-width: 100%;
  }

  .tl-step {
    display: flex;
    flex-direction: column;
    gap: 3px;
    flex-basis: 0;
    min-width: 3px;
    overflow: hidden;
  }

  .tl-step-head {
    display: flex;
    align-items: baseline;
    gap: 6px;
    font-size: 10px;
    line-height: 14px;
    white-space: nowrap;
    overflow: hidden;
  }

  .tl-step-name {
    color: var(--text-bright);
    font-weight: 500;
  }

  .tl-step-dur {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .tl-bar {
    height: 18px;
  }

  .tl-hint {
    font-size: 10px;
    color: var(--muted);
    opacity: 0.7;
  }

  /* ── Pinnable tooltip ────────────────────────────────────────────────── */
  .tg-tip {
    position: fixed;
    z-index: 1000;
    transform: translate(-50%, calc(-100% - 10px));
    pointer-events: none;
    display: flex;
    flex-direction: column;
    gap: 1px;
    padding: 6px 9px;
    border-radius: 6px;
    background: var(--color-c-gray-02, #0d0d0d);
    border: 1px solid var(--border, #3a3a3a);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
    font-size: 11px;
    white-space: nowrap;
  }

  .tg-tip.pinned {
    border-color: var(--accent, #60a5fa);
  }

  .tg-tip-step {
    color: var(--muted);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .tg-tip-name {
    color: var(--text-bright, #fff);
    font-weight: 600;
  }

  .tg-tip-dur {
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .tg-tip-rank {
    color: var(--muted);
    font-size: 10px;
    font-variant-numeric: tabular-nums;
  }

  .tg-tip-summary {
    color: var(--text);
    margin-top: 4px;
  }

  .tg-tip-description {
    max-width: 360px;
    margin-top: 4px;
    color: var(--muted);
    white-space: normal;
  }

  .tg-tip-profile {
    display: grid;
    grid-template-columns: auto auto;
    column-gap: 12px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .tg-tip-profile span:nth-child(even) {
    text-align: right;
  }
</style>
