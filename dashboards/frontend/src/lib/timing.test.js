import assert from "node:assert/strict";
import test from "node:test";

import {
  anchorLanes,
  clipIdleSpans,
  groupTooltipChildren,
  HIDDEN_PHASES,
  timingIsAsync,
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
  }, false);
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

const generationPayload = (sync) => {
  const phase = (total, start, end, count = 1, invocations = [[start, end]]) => ({
    count,
    total_duration_s: total,
    first_start_s: start,
    last_end_s: end,
    invocations,
  });
  const rollout = {
    lane_start_unix_s: 100,
    phases: {
      generate_samples: phase(10, 0, 10),
      sample_generation: phase(2, 1, 3),
      reward: phase(1, 3, 4),
      reward_post_process: phase(2, 4, 6, 2, [
        [4, 5],
        [5, 6],
      ]),
    },
  };
  return {
    0: {
      roles: {
        driver: {
          lane_start_unix_s: 100,
          phases: sync
            ? { generate_rollouts: phase(10, 0, 10) }
            : { train_models: phase(10, 10, 20) },
        },
        rollout,
      },
    },
  };
};

test("sync generation tooltips preserve the async phase breakdown", () => {
  const syncTimeline = runTimeline(generationPayload(true));
  const asyncTimeline = runTimeline(generationPayload(false));
  const barFor = (timeline, name) =>
    timeline.groups
      .flatMap((group) => group.rows.flatMap((row) => row.spans))
      .find((span) => span.name === name);
  const tooltipChildren = (bar) =>
    groupTooltipChildren(bar.children, bar.aggregateStats).map(
      ({ name, duration, count }) => ({ name, duration, count }),
    );

  const syncGeneration = barFor(syncTimeline, "generate_rollouts");
  const asyncGeneration = barFor(asyncTimeline, "generate_samples");
  assert.equal(syncGeneration.children.length, 0);
  assert.deepEqual(
    tooltipChildren(syncGeneration),
    tooltipChildren(asyncGeneration),
  );
  assert.deepEqual(tooltipChildren(syncGeneration), [
    {
      name: "reward_post_process",
      duration: 2,
      count: 2,
    },
  ]);

  const drawnSyncBars = syncTimeline.groups
    .flatMap((group) => group.rows.flatMap((row) => row.spans))
    .filter(
      (span) =>
        span.depth === 0 &&
        !span.mergedGeneration &&
        !HIDDEN_PHASES.has(span.name),
    );
  assert.equal(drawnSyncBars.length, 1);
});

test("rollout spans get their own row without overlap detection", () => {
  const payload = generationPayload(false);
  const timeline = runTimeline(payload, timingIsAsync(payload));
  assert.equal(timeline.async, true);
  assert.ok(
    timeline.groups[0].rows.some(
      (row) => row.role === "rollout" && row.spans.some((span) => span.role === "rollout"),
    ),
  );
  assert.ok(
    !timeline.groups[0].rows
      .find((row) => row.key === "driver")
      .spans.some((span) => span.role === "rollout"),
  );
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

test("anchorLanes centers a substantially shorter shifted lane", () => {
  const spans = anchoredLane(10, 20, 20, 25);
  anchorLanes(spans);

  assert.deepEqual(
    [spans[1].start, spans[1].end],
    [12.5, 17.5],
  );
  assert.ok(Math.abs(spans[1].clockOffset + 7.5) < 1e-9);
  assert.equal(spans[1].clockShifted, true);
});

test("anchorLanes leaves an already-fitting lane unchanged", () => {
  const spans = anchoredLane(10, 20, 12, 18);
  anchorLanes(spans);

  assert.deepEqual([spans[1].start, spans[1].end], [12, 18]);
  assert.equal(spans[1].clockShifted, undefined);
  assert.equal(spans[1].clockOffset, undefined);
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
    end: 165,
  };

  anchorLanes([driverA, driverB, actor]);
  nest([driverA, driverB, actor]);

  assert.deepEqual([actor.start, actor.end], [149.99, 179.99]);
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

test("runTimeline keeps multiple worker roles nested under one parent", () => {
  const timeline = runTimeline({
    0: {
      roles: {
        driver: {
          lane_start_unix_s: 100,
          phases: {
            train_models: {
              count: 1,
              total_duration_s: 20,
              first_start_s: 0,
              last_end_s: 20,
            },
          },
        },
        actor: {
          lane_start_unix_s: 100,
          phases: {
            forward_backward: {
              count: 1,
              total_duration_s: 4,
              first_start_s: 2,
              last_end_s: 6,
            },
          },
        },
        critic: {
          lane_start_unix_s: 100,
          phases: {
            forward_backward: {
              count: 1,
              total_duration_s: 4,
              first_start_s: 2,
              last_end_s: 6,
            },
          },
        },
      },
    },
  });
  const rows = timeline.groups[0].rows;
  const train = rows[0].spans.find(
    (span) => span.name === "train_models" && span.depth === 0,
  );
  const roles = new Map();
  for (const child of train.children) {
    const group = roles.get(child.role) || [];
    group.push(child);
    roles.set(child.role, group);
  }

  assert.equal(rows.length, 1);
  assert.deepEqual([...roles.keys()], ["actor", "critic"]);
  assert.ok([...roles.values()].every((children) => children.length === 1));
  assert.deepEqual(
    [...roles.values()].map(([child]) => [child.start, child.end]),
    [
      [102, 106],
      [102, 106],
    ],
  );
  assert.notEqual(roles.get("actor")[0], roles.get("critic")[0]);
});

test("runTimeline keeps a single worker role layout unchanged", () => {
  const timeline = runTimeline({
    0: {
      roles: {
        driver: {
          lane_start_unix_s: 100,
          phases: {
            train_models: {
              count: 1,
              total_duration_s: 20,
              first_start_s: 0,
              last_end_s: 20,
            },
          },
        },
        actor: {
          lane_start_unix_s: 100,
          phases: {
            forward_backward: {
              count: 1,
              total_duration_s: 4,
              first_start_s: 2,
              last_end_s: 6,
            },
          },
        },
      },
    },
  });
  const train = timeline.groups[0].rows[0].spans.find(
    (span) => span.name === "train_models",
  );

  assert.equal(timeline.groups[0].rows.length, 1);
  assert.deepEqual(
    [train.children[0].start, train.children[0].end],
    [102, 106],
  );
  assert.equal(train.children[0].role, "actor");
});

test("runTimeline keeps rows available when details are collapsed", () => {
  const timeline = runTimeline({
    0: {
      roles: {
        driver: {
          lane_start_unix_s: 100,
          phases: {
            train_models: {
              count: 1,
              total_duration_s: 20,
              first_start_s: 0,
              last_end_s: 20,
            },
            checkpoint_save: {
              count: 1,
              total_duration_s: 2,
              first_start_s: 20,
              last_end_s: 22,
            },
          },
        },
      },
    },
  });
  const [group] = timeline.groups;

  assert.ok(group.rows.length > 0);
  assert.ok(
    group.rows.some((row) =>
      row.spans.some((span) => span.depth === 0),
    ),
  );
});
