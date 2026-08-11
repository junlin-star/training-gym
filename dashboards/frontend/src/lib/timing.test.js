import assert from "node:assert/strict";
import test from "node:test";

import { anchorLanes, clipIdleSpans, nest, runTimeline } from "./timing.js";

const work = (start, end) => ({
  kind: "work",
  role: "driver",
  start,
  end,
  name: "train_models",
  rolloutId: 1,
});

const idle = (start, end) => ({
  kind: "idle",
  role: "driver",
  start,
  end,
  name: "wait_for_rollout",
  rolloutId: 1,
});

test("clipIdleSpans merges work ranges before clipping", () => {
  const spans = [
    idle(0, 10),
    work(1, 2),
    work(3, 5),
    work(4, 6),
    work(8, 9),
    idle(1, 2),
  ];
  const clipped = clipIdleSpans(spans, false)
    .filter((span) => span.kind === "idle")
    .map(({ start, end }) => [start, end]);

  assert.deepEqual(clipped, [
    [0, 1],
    [2, 3],
    [6, 8],
    [9, 10],
  ]);
});

test("clipIdleSpans rescans overlapping idle spans from their own starts", () => {
  const clipped = clipIdleSpans(
    [idle(0, 10), idle(1, 2), work(1, 2)],
    false,
  )
    .filter((span) => span.kind === "idle")
    .map(({ start, end }) => [start, end]);

  assert.deepEqual(clipped, [
    [0, 1],
    [2, 10],
  ]);
});

test("nest tolerates small cross-lane clock skew only", () => {
  const parent = {
    name: "train_models",
    rolloutId: 1,
    group: "step",
    role: "driver",
    laneKey: "a",
    start: 10,
    end: 20,
  };
  const child = {
    name: "forward_backward",
    rolloutId: 1,
    group: "step",
    role: "actor",
    laneKey: "b",
    start: 9.998,
    end: 20.002,
  };
  const outside = {
    ...child,
    laneKey: "c",
    start: 9.98,
    end: 20.02,
  };

  const nested = nest([parent, child, outside]);
  assert.equal(nested.find((span) => span === child).parent, parent);
  assert.equal(nested.find((span) => span === outside).parent, null);
});

test("runTimeline keeps adjacent generation outside the train parent", () => {
  const timeline = runTimeline({
    0: {
      roles: {
        driver: {
          lane_start_unix_s: 100,
          phases: {
            train_models: {
              count: 1,
              total_duration_s: 5,
              first_start_s: 5,
              last_end_s: 10,
            },
          },
        },
        rollout: {
          lane_start_unix_s: 100,
          phases: {
            generate_samples: {
              count: 1,
              total_duration_s: 5,
              first_start_s: 0,
              last_end_s: 5,
            },
          },
        },
      },
    },
  });
  const roots = timeline.groups[0].rows[0].spans.filter(
    (span) => span.depth === 0,
  );
  const train = roots.find((span) => span.name === "train_models");
  const generation = roots.find((span) => span.name === "generate_samples");

  assert.deepEqual([train.start, train.end], [105, 110]);
  assert.equal(train.parent, undefined);
  assert.equal(generation.parent, undefined);
  assert.ok(generation.end <= train.start);
});

const anchoredLane = (parentStart, parentEnd, laneStart, laneEnd) => [
  {
    name: "train_models",
    category: "train",
    role: "driver",
    rolloutId: 1,
    start: parentStart,
    end: parentEnd,
  },
  {
    name: "forward_backward",
    category: "train",
    role: "actor",
    rolloutId: 1,
    laneKey: "1:actor",
    start: laneStart,
    end: laneEnd,
  },
];

test("anchorLanes shifts a lane just enough to fit its owner", () => {
  const spans = anchoredLane(10, 20, 15, 25);
  anchorLanes(spans);

  assert.deepEqual(
    [spans[1].start, spans[1].end, spans[1].clockOffset],
    [10, 20, -5],
  );
  assert.equal(spans[1].clockShifted, true);
});

test("anchorLanes clamps a shift that would move the lane before its owner", () => {
  const spans = anchoredLane(10, 20, 5, 25);
  anchorLanes(spans);

  assert.deepEqual(
    [spans[1].start, spans[1].end, spans[1].clockOffset],
    [10, 30, 5],
  );
});

test("anchorLanes aligns an overlong lane at the owner start", () => {
  const spans = anchoredLane(10, 20, 0, 30);
  anchorLanes(spans);

  assert.deepEqual(
    [spans[1].start, spans[1].end, spans[1].clockOffset],
    [10, 40, 10],
  );
});

test("anchorLanes shifts every span in a lane without changing durations", () => {
  const spans = [
    ...anchoredLane(10, 20, 15, 25),
    {
      name: "optimizer_step",
      category: "train",
      role: "actor",
      rolloutId: 1,
      laneKey: "1:actor",
      start: 18,
      end: 19,
    },
  ];
  const durations = spans.slice(1).map((span) => span.end - span.start);
  anchorLanes(spans);

  assert.deepEqual(
    spans.slice(1).map((span) => [span.start, span.end]),
    [
      [10, 20],
      [13, 14],
    ],
  );
  assert.deepEqual(
    spans.slice(1).map((span) => span.end - span.start),
    durations,
  );
});
