import assert from "node:assert/strict";
import test from "node:test";

import { phaseBreakdown } from "./asyncTimingProfile.js";

test("phaseBreakdown profiles arbitrary nested phases without double-counting overlap", () => {
  const parent = { name: "training", start: 10, duration: 10 };
  const timeline = [
    interval("forward_backward", "training", "actor", 9, 16),
    interval("forward_backward", "training", "critic", 12, 17),
    interval("data_preprocess", "training", null, 17, 19),
    interval("checkpoint", "coordination", null, 11, 18),
  ];

  const breakdown = phaseBreakdown(timeline, parent);

  assert.equal(breakdown.total, 10);
  assert.equal(breakdown.measured, 9);
  assert.equal(breakdown.other, 1);
  assert.deepEqual(
    breakdown.phases.map(({ key, duration }) => ({ key, duration })),
    [
      { key: "forward_backward:actor", duration: 6 },
      { key: "forward_backward:critic", duration: 5 },
      { key: "data_preprocess:", duration: 2 },
    ],
  );
});

test("phaseBreakdown returns no profile when a phase has no reported children", () => {
  assert.equal(phaseBreakdown([], { name: "training", start: 10, duration: 2 }), null);
});

function interval(name, parentPhase, role, start, end) {
  return {
    sub: { name, parentPhase, role },
    start,
    end,
  };
}
