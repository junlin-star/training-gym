import assert from "node:assert/strict";
import test from "node:test";

import { runTimeline } from "./timing.js";

function lane(role, phases, laneStart = 0) {
  return { role, lane_start_unix_s: laneStart, phases };
}

function phase(start, end, invocations = [[start, end]]) {
  return {
    count: invocations.length,
    total_duration_s: invocations.reduce((total, run) => total + run[1] - run[0], 0),
    longest_duration_s: Math.max(...invocations.map((run) => run[1] - run[0])),
    first_start_s: start,
    last_end_s: end,
    invocations,
  };
}

test("sync timing stays on one row despite rollout-worker records", () => {
  const timeline = runTimeline({
    0: {
      roles: {
        driver: lane("driver", {
          generate_rollouts: phase(0, 10),
          train_models: phase(10, 20),
        }),
        rollout: lane("rollout", {
          generate_samples: phase(0, 10),
        }),
      },
    },
  });

  assert.equal(timeline.async, false);
  assert.equal(timeline.groups[0].rows.length, 1);
});

test("async timing separates overlapping rollout rows", () => {
  const timeline = runTimeline({
    0: {
      roles: {
        driver: lane("driver", {
          train_models: phase(2, 8),
        }),
        rollout: lane("rollout", {
          generate_samples: phase(0, 4),
        }),
      },
    },
  });

  assert.equal(timeline.async, true);
  assert.equal(timeline.groups[0].rows.length, 2);
});
