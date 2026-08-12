import assert from "node:assert/strict";
import test from "node:test";

import {
  anchorLanes,
  clipIdleSpans,
  groupTooltipChildren,
  nest,
  runTimeline,
  shouldShowTimingSection,
} from "./timing.js";

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

test("shouldShowTimingSection includes stale timing without lanes", () => {
  assert.equal(
    shouldShowTimingSection({ metadata: { timing_stale: true } }),
    true,
  );
  assert.equal(shouldShowTimingSection({ metadata: { timing_stale: false } }), false);
  assert.equal(
    shouldShowTimingSection({ metadata: { legacy_derived: true, timing_stale: true } }),
    false,
  );
});

test("groupTooltipChildren preserves order and aggregates repeated phases", () => {
  const children = [
    {
      name: "forward_backward",
      duration: 1.5,
      count: 1,
      start: 0,
      end: 1.5,
    },
    {
      name: "optimizer_step",
      duration: 0.25,
      count: 1,
      start: 1.5,
      end: 1.75,
    },
    {
      name: "forward_backward",
      duration: 1.25,
      count: 1,
      start: 2,
      end: 3.25,
    },
    {
      name: "hidden",
      duration: 4,
      count: 1,
      mergedGeneration: true,
    },
    {
      name: "wait_for_rollout",
      rolloutId: 3,
      duration: 0.5,
      count: 1,
      start: 3.25,
      end: 3.75,
    },
  ];

  assert.deepEqual(
    groupTooltipChildren(children).map(
      ({ name, label, duration, count, representative }) => ({
        name,
        label,
        duration,
        count,
        representative,
      }),
    ),
    [
      {
        name: "forward_backward",
        label: "Forward/backward",
        duration: 2.75,
        count: 2,
        representative: children[0],
      },
      {
        name: "optimizer_step",
        label: "Optimizer step",
        duration: 0.25,
        count: 1,
        representative: children[1],
      },
      {
        name: "wait_for_rollout",
        label: "Waiting for rollout generation (step 3)",
        duration: 0.5,
        count: 1,
        representative: children[4],
      },
    ],
  );
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

test("anchorLanes chooses a concrete parent instead of a disjoint union", () => {
  const driverA = {
    name: "train_models",
    category: "train",
    role: "driver",
    rolloutId: 1,
    start: 100,
    end: 130,
  };
  const driverB = {
    ...driverA,
    start: 150,
    end: 180,
  };
  const actor = {
    name: "forward_backward",
    category: "train",
    role: "actor",
    rolloutId: 1,
    laneKey: "1:actor",
    start: 135,
    end: 155,
  };

  anchorLanes([driverA, driverB, actor]);
  nest([driverA, driverB, actor]);

  assert.deepEqual([actor.start, actor.end], [149.99, 169.99]);
  assert.ok(Math.abs(actor.clockOffset - 14.99) < 1e-9);
  assert.equal(actor.depth, 1);
  assert.equal(actor.parent, driverB);
});

test("anchorLanes leaves a lane unshifted when no concrete parent fits", () => {
  const spans = [
    {
      name: "train_models",
      category: "train",
      role: "driver",
      rolloutId: 1,
      start: 100,
      end: 110,
    },
    {
      name: "train_models",
      category: "train",
      role: "driver",
      rolloutId: 1,
      start: 120,
      end: 130,
    },
    {
      name: "forward_backward",
      category: "train",
      role: "actor",
      rolloutId: 1,
      laneKey: "1:actor",
      start: 90,
      end: 130,
    },
  ];

  anchorLanes(spans);

  assert.deepEqual([spans[2].start, spans[2].end], [90, 130]);
  assert.equal(spans[2].clockShifted, undefined);
  assert.equal(spans[2].clockOffset, undefined);
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
