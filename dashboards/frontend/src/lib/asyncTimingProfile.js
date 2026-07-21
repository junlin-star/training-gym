function coveredDuration(intervals) {
  const ordered = [...intervals].sort((a, b) => a.start - b.start);
  let covered = 0;
  let currentStart = null;
  let currentEnd = null;
  for (const interval of ordered) {
    if (currentEnd == null || interval.start > currentEnd) {
      if (currentEnd != null) covered += currentEnd - currentStart;
      currentStart = interval.start;
      currentEnd = interval.end;
    } else {
      currentEnd = Math.max(currentEnd, interval.end);
    }
  }
  if (currentEnd != null) covered += currentEnd - currentStart;
  return covered;
}

export function phaseBreakdown(timeline, parent) {
  const total = Number(parent.duration);
  const parentStart = Number(parent.start);
  if (!Number.isFinite(total) || total <= 0 || !Number.isFinite(parentStart)) return null;
  const parentEnd = parentStart + total;
  const groups = new Map();
  const coveredIntervals = [];

  for (const interval of timeline) {
    if (interval.sub.parentPhase !== parent.name) continue;
    const start = Math.max(parentStart, interval.start);
    const end = Math.min(parentEnd, interval.end);
    if (end <= start) continue;
    const role = interval.sub.trainingRole || null;
    const key = `${interval.sub.name}:${role || ""}`;
    if (!groups.has(key)) {
      groups.set(key, { key, phase: interval.sub, role, intervals: [], start });
    }
    const clipped = { start, end };
    groups.get(key).intervals.push(clipped);
    coveredIntervals.push(clipped);
  }
  if (!groups.size) return null;

  const phases = [...groups.values()]
    .sort((a, b) => a.start - b.start)
    .map((group) => ({
      key: group.key,
      phase: group.phase,
      role: group.role,
      duration: coveredDuration(group.intervals),
    }));
  const measured = Math.min(total, coveredDuration(coveredIntervals));
  return { total, measured, other: Math.max(0, total - measured), phases };
}
