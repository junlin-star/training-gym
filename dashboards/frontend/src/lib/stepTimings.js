export function mergeStepTimes(
  previousSteps,
  fetchedSteps,
  preferPrevious = false,
) {
  if (!previousSteps) return fetchedSteps || null;
  if (!fetchedSteps) return previousSteps;
  const merged = { ...previousSteps };
  for (const [step, timing] of Object.entries(fetchedSteps)) {
    const mergedTiming = { ...(merged[step] || {}) };
    for (const [key, value] of Object.entries(timing || {})) {
      if (
        !(key in mergedTiming) ||
        mergedTiming[key] == null ||
        (!preferPrevious && value != null)
      ) {
        mergedTiming[key] = value;
      }
    }
    merged[step] = mergedTiming;
  }
  return merged;
}

export function mergeSubstepTimes(
  previousSubsteps,
  fetchedSubsteps,
  preferPrevious = false,
) {
  if (!previousSubsteps) return fetchedSubsteps || null;
  if (!fetchedSubsteps) return previousSubsteps;
  const merged = { ...previousSubsteps };
  for (const [step, timings] of Object.entries(fetchedSubsteps)) {
    const mergedStep = { ...(merged[step] || {}) };
    for (const [phase, timing] of Object.entries(timings || {})) {
      const previousTiming = mergedStep[phase];
      const previousIntervals = previousTiming?.intervals?.length || 0;
      const fetchedIntervals = timing?.intervals?.length || 0;
      const previousHasDuration = previousTiming?.duration_s != null;
      const fetchedHasDuration = timing?.duration_s != null;
      if (
        previousIntervals < fetchedIntervals ||
        (previousIntervals === fetchedIntervals &&
          (!previousHasDuration || (!preferPrevious && fetchedHasDuration)))
      ) {
        mergedStep[phase] = timing;
      }
    }
    merged[step] = mergedStep;
  }
  return merged;
}
