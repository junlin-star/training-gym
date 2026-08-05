import { describe, expect, it } from "vitest";
import { colorFor, fmtSecs, labelFor, rolloutTimeline, TIMING_COLORS, TIMING_LABELS } from "./timing.js";

describe("labelFor", () => {
  it("returns a known label", () => {
    expect(labelFor("generate_rollouts")).toBe("Generate rollouts");
  });

  it("falls back to title-casing an unknown name", () => {
    expect(labelFor("unknown_phase")).toBe("unknown phase");
  });
});

describe("colorFor", () => {
  it("returns the color for a known phase", () => {
    expect(colorFor("generate_rollouts")).toBe(TIMING_COLORS.generate_rollouts);
  });

  it("returns the fallback for an unknown phase", () => {
    expect(colorFor("weird_phase")).toBe("var(--color-c-gray-40, #5e5e5e)");
  });
});

describe("fmtSecs", () => {
  it("formats seconds with ms", () => {
    expect(fmtSecs(1.2346)).toBe("1.235s");
  });

  it("formats minutes when over 60s", () => {
    expect(fmtSecs(90.5)).toBe("1m 30.5s");
  });

  it("handles null/undefined", () => {
    expect(fmtSecs(null)).toBe("—");
    expect(fmtSecs(undefined)).toBe("—");
  });
});

/** A phase from its runs, as the recorder accumulates them. */
const phase = (runs) => ({
  count: runs.length,
  total_duration_s: runs.reduce((acc, [start, end]) => acc + (end - start), 0),
  longest_duration_s: Math.max(...runs.map(([start, end]) => end - start)),
  first_start_s: Math.min(...runs.map(([start]) => start)),
  last_end_s: Math.max(...runs.map(([, end]) => end)),
  invocations: runs,
});

/** A per-sample phase, which recorded its aggregate but not its runs. */
const perSamplePhase = (count, total, longest, start, end) => ({
  count,
  total_duration_s: total,
  longest_duration_s: longest,
  first_start_s: start,
  last_end_s: end,
  invocations: [],
});

describe("rolloutTimeline", () => {
  it("draws one bar per run, so alternating phases do not nest", () => {
    const { rows, span } = rolloutTimeline({
      roles: {
        actor: {
          role: "actor",
          lane_start_unix_s: 1000,
          phases: {
            forward_backward: phase([
              [0, 1],
              [2, 3],
            ]),
            optimizer_step: phase([
              [1, 2],
              [3, 4],
            ]),
          },
        },
      },
    });

    // Nothing overlapped, so all four bars fit on one row in the order they ran.
    expect(rows).toHaveLength(1);
    expect(rows[0].bars.map((bar) => bar.name)).toEqual([
      "forward_backward",
      "optimizer_step",
      "forward_backward",
      "optimizer_step",
    ]);
    expect(rows[0].bars.map((bar) => bar.duration)).toEqual([1, 1, 1, 1]);
    expect(span).toBe(4);
  });

  it("puts a bar on a second row only where work really overlapped", () => {
    const { rows } = rolloutTimeline({
      roles: {
        rollout: {
          role: "rollout",
          lane_start_unix_s: 1000,
          phases: {
            generate_samples: phase([
              [0, 4],
              [5, 9],
            ]),
            wait_for_rollout: phase([[3, 6]]),
          },
        },
      },
    });

    expect(rows).toHaveLength(2);
    expect(rows[0].concurrent).toBe(false);
    expect(rows[1].concurrent).toBe(true);
    expect(rows[1].bars.map((bar) => bar.name)).toEqual(["wait_for_rollout"]);
  });

  it("nests a run that happened inside another instead of giving it a row", () => {
    const { rows } = rolloutTimeline({
      roles: {
        driver: {
          role: "driver",
          lane_start_unix_s: 1000,
          phases: { train_models: phase([[0, 4]]) },
        },
        actor: {
          role: "actor",
          lane_start_unix_s: 1000,
          phases: {
            forward_backward: phase([
              [0.1, 1.5],
              [2, 3.4],
            ]),
            optimizer_step: phase([
              [1.5, 2],
              [3.4, 3.9],
            ]),
          },
        },
      },
    });

    // The step, then the four runs it contains on one row drawn within it.
    expect(rows).toHaveLength(2);
    expect(rows[0].bars.map((bar) => [bar.name, bar.depth])).toEqual([
      ["train_models", 0],
    ]);
    expect([rows[1].depth, rows[1].concurrent]).toEqual([1, false]);
    expect(rows[1].bars.map((bar) => bar.name)).toEqual([
      "forward_backward",
      "optimizer_step",
      "forward_backward",
      "optimizer_step",
    ]);
  });

  it("shifts lanes recorded in different processes onto one axis", () => {
    const { rows } = rolloutTimeline({
      roles: {
        driver: {
          role: "driver",
          lane_start_unix_s: 1000,
          phases: { train_models: phase([[0, 1]]) },
        },
        actor: {
          role: "actor",
          lane_start_unix_s: 1002,
          phases: { forward_backward: phase([[0, 0.5]]) },
        },
      },
    });

    // The actor's lane opened 2s after the driver's, so its bar sits there and
    // does not collide with the driver's first second.
    expect(rows).toHaveLength(1);
    expect(rows[0].bars.map((bar) => [bar.name, bar.start])).toEqual([
      ["train_models", 0],
      ["forward_backward", 2],
    ]);
  });

  it("has nothing to draw for a rollout that recorded no timing", () => {
    expect(rolloutTimeline({ roles: {} }).rows).toEqual([]);
    expect(rolloutTimeline(null).rows).toEqual([]);
  });

  it("reads a per-sample phase as work spent in the phase that contains it", () => {
    const { rows } = rolloutTimeline({
      roles: {
        rollout: {
          role: "rollout",
          lane_start_unix_s: 1000,
          phases: {
            generate_samples: phase([[0, 4]]),
            reward: perSamplePhase(64, 0.64, 0.05, 0.5, 3.9),
          },
        },
      },
    });

    // Its calls covered most of the generation, but the generation is the bar:
    // a block over that span would read as reward working the whole time.
    const bars = rows.flatMap((row) => row.bars);
    expect(bars.map((bar) => bar.name)).toEqual(["generate_samples"]);
    expect(bars[0].spent).toEqual([
      {
        role: "rollout",
        name: "reward",
        count: 64,
        total: 0.64,
        longest: 0.05,
        start: 0.5,
        end: 3.9,
      },
    ]);
  });

  it("draws a per-sample phase nothing contains rather than dropping it", () => {
    const { rows } = rolloutTimeline({
      roles: {
        rollout: {
          role: "rollout",
          lane_start_unix_s: 1000,
          phases: { reward: perSamplePhase(64, 0.64, 0.05, 0.5, 3.9) },
        },
      },
    });

    const bars = rows.flatMap((row) => row.bars);
    expect(bars.map((bar) => [bar.name, bar.duration])).toEqual([["reward", 0.64]]);
  });

  it("leaves out a phase whose runs did no measurable work", () => {
    const { rows } = rolloutTimeline({
      roles: {
        rollout: {
          role: "rollout",
          lane_start_unix_s: 1000,
          phases: {
            generate_samples: phase([[0, 4]]),
            reward: phase([
              [1, 1.00001],
              [2, 2.00001],
            ]),
          },
        },
      },
    });

    expect(rows.flatMap((row) => row.bars).map((bar) => bar.name)).toEqual([
      "generate_samples",
    ]);
  });

  it("reads a phase whose every run is too short to see as work spent", () => {
    const scored = Array.from({ length: 64 }, (_, index) => [
      1 + index * 0.01,
      1 + index * 0.01 + 0.0001,
    ]);
    const { rows } = rolloutTimeline({
      roles: {
        rollout: {
          role: "rollout",
          lane_start_unix_s: 1000,
          phases: { generate_samples: phase([[0, 4]]), reward: phase(scored) },
        },
      },
    });

    // The runs are kept in the record, but 64 slivers of 0.1ms across the
    // generation are the generation's reward cost, not 64 bars.
    const bars = rows.flatMap((row) => row.bars);
    expect(bars.map((bar) => bar.name)).toEqual(["generate_samples"]);
    expect(bars[0].spent[0].count).toBe(64);
  });

  it("draws runs of one phase that overlapped as the block of work they formed", () => {
    const { rows } = rolloutTimeline({
      roles: {
        rollout: {
          role: "rollout",
          lane_start_unix_s: 1000,
          phases: {
            reward: phase([
              [0, 2],
              [1, 3],
              [5, 6],
            ]),
          },
        },
      },
    });

    const bars = rows.flatMap((row) => row.bars);
    expect(bars.map((bar) => [bar.start, bar.end, bar.runs])).toEqual([
      [0, 3, 2],
      [5, 6, 1],
    ]);
    expect(bars[0].work).toBe(4);
  });

  it("keeps a checkpoint out of the step's duration but still draws it", () => {
    const { stepDuration, beside, rows } = rolloutTimeline({
      roles: {
        driver: {
          role: "driver",
          lane_start_unix_s: 1000,
          phases: {
            train_models: phase([[0, 2]]),
            checkpoint_save: phase([[2, 12]]),
          },
        },
      },
    });

    expect(stepDuration).toBe(2);
    expect(beside).toEqual([{ name: "checkpoint_save", duration: 10 }]);
    expect(rows.flatMap((row) => row.bars).map((bar) => bar.name)).toContain(
      "checkpoint_save",
    );
  });
});

it("says what a bar ran inside and what it ran alongside", () => {
  const { rows } = rolloutTimeline({
    roles: {
      driver: {
        role: "driver",
        lane_start_unix_s: 1000,
        phases: { train_models: phase([[0, 4]]) },
      },
      actor: {
        role: "actor",
        lane_start_unix_s: 1000,
        phases: { forward_backward: phase([[1, 2]]) },
      },
      rollout: {
        role: "rollout",
        lane_start_unix_s: 1000,
        phases: { generate_samples: phase([[3, 6]]) },
      },
    },
  });

  const bars = rows.flatMap((row) => row.bars);
  const inside = bars.find((bar) => bar.name === "forward_backward");
  const alongside = bars.find((bar) => bar.name === "generate_samples");
  expect(inside.inside).toBe("train_models");
  expect(inside.overlaps).toEqual([]);
  expect(alongside.inside).toBe(null);
  expect(alongside.overlaps).toEqual(["train_models"]);
});
