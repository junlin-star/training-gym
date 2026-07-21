import assert from "node:assert/strict";
import test from "node:test";

import { phaseBreakdown } from "./asyncTimingProfile.js";

test("phaseBreakdown profiles arbitrary nested phases without double-counting overlap", () => {
  const parent = { name: "data_preparation", start: 10, duration: 10 };
  const timeline = [
    interval("policy_evaluation", "data_preparation", "actor", 9, 16),
    interval("policy_evaluation", "data_preparation", "critic", 12, 17),
    interval("batch_packing", "data_preparation", null, 17, 19),
    interval("checkpoint", "coordination", null, 11, 18),
  ];

  const breakdown = phaseBreakdown(timeline, parent);

  assert.equal(breakdown.total, 10);
  assert.equal(breakdown.measured, 9);
  assert.equal(breakdown.other, 1);
  assert.deepEqual(
    breakdown.phases.map(({ key, duration }) => ({ key, duration })),
    [
      { key: "policy_evaluation:actor", duration: 6 },
      { key: "policy_evaluation:critic", duration: 5 },
      { key: "batch_packing:", duration: 2 },
    ],
  );
});

test("phaseBreakdown returns no profile when a phase has no reported children", () => {
  assert.equal(phaseBreakdown([], { name: "training", start: 10, duration: 2 }), null);
});

function interval(name, parentPhase, trainingRole, start, end) {
  return {
    sub: { name, parentPhase, trainingRole },
    start,
    end,
  };
}
