<script>
  import { Calendar, ChevronLeft, ChevronRight } from "lucide-svelte";

  // `value` is epoch seconds (number) or null. Bindable so the parent can both
  // seed it (run start/end) and read edits. We deliberately avoid the native
  // <input type="datetime-local"> because its popup is drawn by the browser and
  // can't be themed (the highlight is always browser-blue) — this is a fully
  // CSS-styled replacement.
  let {
    value = $bindable(null),
    placeholder = "select…",
    ariaLabel = "",
  } = $props();

  const MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ];
  const WEEKDAYS = ["S", "M", "T", "W", "T", "F", "S"];
  const pad = (x) => String(x).padStart(2, "0");

  let open = $state(false);
  let rootEl = $state(null);
  // The month shown in the grid (its day is irrelevant).
  let viewDate = $state(new Date());

  let selected = $derived(
    value != null && value !== "" ? new Date(Number(value) * 1000) : null,
  );

  let triggerLabel = $derived.by(() => {
    if (!selected) return "";
    return (
      `${MONTHS[selected.getMonth()]} ${selected.getDate()}, ` +
      `${selected.getFullYear()} ${pad(selected.getHours())}:${pad(selected.getMinutes())}`
    );
  });

  // 6-week (null-padded) grid of day numbers for the viewed month.
  let cells = $derived.by(() => {
    const y = viewDate.getFullYear();
    const m = viewDate.getMonth();
    const startDow = new Date(y, m, 1).getDay();
    const daysInMonth = new Date(y, m + 1, 0).getDate();
    const out = [];
    for (let i = 0; i < startDow; i++) out.push(null);
    for (let d = 1; d <= daysInMonth; d++) out.push(d);
    while (out.length % 7 !== 0) out.push(null);
    return out;
  });

  const today = new Date();
  function isToday(day) {
    return (
      day != null &&
      viewDate.getFullYear() === today.getFullYear() &&
      viewDate.getMonth() === today.getMonth() &&
      day === today.getDate()
    );
  }
  function isSelected(day) {
    return (
      day != null &&
      selected != null &&
      selected.getFullYear() === viewDate.getFullYear() &&
      selected.getMonth() === viewDate.getMonth() &&
      selected.getDate() === day
    );
  }

  function commit(y, m, d, h, min) {
    const dt = new Date(y, m, d, h, min, 0, 0);
    value = Math.floor(dt.getTime() / 1000);
  }

  function pickDay(day) {
    if (day == null) return;
    const cur = selected || new Date();
    commit(viewDate.getFullYear(), viewDate.getMonth(), day, cur.getHours(), cur.getMinutes());
  }

  function onHour(e) {
    const h = Math.max(0, Math.min(23, parseInt(e.currentTarget.value, 10) || 0));
    const cur = selected || new Date();
    commit(cur.getFullYear(), cur.getMonth(), cur.getDate(), h, cur.getMinutes());
  }
  function onMinute(e) {
    const min = Math.max(0, Math.min(59, parseInt(e.currentTarget.value, 10) || 0));
    const cur = selected || new Date();
    commit(cur.getFullYear(), cur.getMonth(), cur.getDate(), cur.getHours(), min);
  }

  function stepMonth(delta) {
    viewDate = new Date(viewDate.getFullYear(), viewDate.getMonth() + delta, 1);
  }
  function setNow() {
    const n = new Date();
    viewDate = new Date(n.getFullYear(), n.getMonth(), 1);
    commit(n.getFullYear(), n.getMonth(), n.getDate(), n.getHours(), n.getMinutes());
  }
  function clear() {
    value = null;
  }

  function toggle() {
    open = !open;
    if (open) viewDate = selected ? new Date(selected) : new Date();
  }

  // Close on outside click / Escape while open.
  $effect(() => {
    if (!open) return;
    const onDown = (e) => {
      if (rootEl && !rootEl.contains(e.target)) open = false;
    };
    const onKey = (e) => {
      if (e.key === "Escape") open = false;
    };
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  });
</script>

<div class="relative inline-block" bind:this={rootEl}>
  <button
    type="button"
    class="dtp-trigger inline-flex items-center gap-[6px] bg-(--color-c-gray-10,#1c1c1c) text-(--text) rounded-[5px] p-[5px_8px] text-[12px] [font-family:inherit] cursor-pointer"
    class:dtp-trigger-open={open}
    onclick={toggle}
    aria-label={ariaLabel}
    aria-haspopup="dialog"
    aria-expanded={open}
  >
    <span class={triggerLabel ? "text-(--text)" : "text-(--muted-strong)"}>
      {triggerLabel || placeholder}
    </span>
    <Calendar size={13} class="text-(--muted)" />
  </button>

  {#if open}
    <div
      class="absolute top-[calc(100%+4px)] left-0 z-[40] w-[240px] p-[10px] rounded-[8px] bg-(--panel) [border:1px_solid_var(--border-strong,#464646)] [box-shadow:0_8px_24px_rgba(0,0,0,0.45)]"
      role="dialog"
    >
      <!-- Month nav -->
      <div class="flex items-center justify-between mb-[8px]">
        <button type="button" class="dtp-nav" onclick={() => stepMonth(-1)} aria-label="Previous month">
          <ChevronLeft size={15} />
        </button>
        <span class="text-[12px] font-medium text-(--text-bright)">
          {MONTHS[viewDate.getMonth()]} {viewDate.getFullYear()}
        </span>
        <button type="button" class="dtp-nav" onclick={() => stepMonth(1)} aria-label="Next month">
          <ChevronRight size={15} />
        </button>
      </div>

      <!-- Weekday header -->
      <div class="grid grid-cols-7 gap-[2px] mb-[2px]">
        {#each WEEKDAYS as w, i (i)}
          <span class="text-center text-[10px] text-(--muted) uppercase">{w}</span>
        {/each}
      </div>

      <!-- Day grid -->
      <div class="grid grid-cols-7 gap-[2px]">
        {#each cells as day, i (i)}
          {#if day == null}
            <span></span>
          {:else}
            <button
              type="button"
              class="dtp-day"
              class:dtp-day-selected={isSelected(day)}
              class:dtp-day-today={isToday(day) && !isSelected(day)}
              onclick={() => pickDay(day)}
            >
              {day}
            </button>
          {/if}
        {/each}
      </div>

      <!-- Time -->
      <div class="flex items-center gap-[6px] mt-[10px] pt-[10px] [border-top:1px_solid_var(--border,#2f2f2f)]">
        <span class="text-[10px] text-(--muted) uppercase tracking-[0.04em]">time</span>
        <input
          class="dtp-time"
          type="number"
          min="0"
          max="23"
          value={selected ? pad(selected.getHours()) : "00"}
          oninput={onHour}
          aria-label="Hour"
        />
        <span class="text-(--muted)">:</span>
        <input
          class="dtp-time"
          type="number"
          min="0"
          max="59"
          value={selected ? pad(selected.getMinutes()) : "00"}
          oninput={onMinute}
          aria-label="Minute"
        />
        <div class="ml-auto flex items-center gap-[8px]">
          <button type="button" class="dtp-action" onclick={setNow}>Now</button>
          <button type="button" class="dtp-action" onclick={clear}>Clear</button>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .dtp-trigger {
    border: 1px solid var(--border, #3a3a3a);
  }
  .dtp-trigger:hover {
    border-color: var(--border-strong, #4a4a4a);
  }
  .dtp-trigger-open {
    border-color: var(--accent-border);
  }

  .dtp-nav {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 2px;
    border: 0;
    background: none;
    color: var(--muted);
    cursor: pointer;
    border-radius: 4px;
  }
  .dtp-nav:hover {
    color: var(--text-bright);
    background: var(--color-c-gray-10, #2f2f2f);
  }

  .dtp-day {
    aspect-ratio: 1;
    border: 1px solid transparent;
    background: none;
    color: var(--text);
    font: inherit;
    font-size: 11px;
    border-radius: 5px;
    cursor: pointer;
    font-variant-numeric: tabular-nums;
  }
  .dtp-day:hover {
    background: var(--color-c-gray-10, #2f2f2f);
  }
  .dtp-day-today {
    border-color: var(--border-strong, #4a4a4a);
    color: var(--text-bright);
  }
  .dtp-day-selected {
    background: color-mix(in srgb, var(--accent) 22%, transparent);
    border-color: color-mix(in srgb, var(--accent) 55%, transparent);
    color: var(--accent);
  }

  .dtp-time {
    width: 40px;
    background: var(--color-c-gray-10, #1c1c1c);
    color: var(--text);
    border: 1px solid var(--border, #3a3a3a);
    border-radius: 4px;
    padding: 3px 4px;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
    text-align: center;
    font-family: inherit;
  }
  .dtp-time:focus {
    outline: none;
    border-color: color-mix(in srgb, var(--accent) 55%, transparent);
  }

  .dtp-action {
    border: 0;
    background: none;
    color: var(--muted);
    font: inherit;
    font-size: 11px;
    cursor: pointer;
    padding: 0;
  }
  .dtp-action:hover {
    color: var(--accent);
    text-decoration: underline;
  }
</style>
