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
const bandedPhase = (count, total, longest, start, end) => ({
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
    expect(rows[0].map((bar) => bar.name)).toEqual([
      "forward_backward",
      "optimizer_step",
      "forward_backward",
      "optimizer_step",
    ]);
    expect(rows[0].map((bar) => bar.duration)).toEqual([1, 1, 1, 1]);
    expect(span).toBe(4);
  });

  it("puts a bar on a second row only where work really overlapped", () => {
    const { rows } = rolloutTimeline({
      roles: {
        rollout: {
          role: "rollout",
          lane_start_unix_s: 1000,
          phases: {
            generate_samples: phase([[0, 4]]),
            reward: bandedPhase(32, 6.4, 0.3, 2, 4.5),
          },
        },
      },
    });

    expect(rows).toHaveLength(2);
    expect(rows[0].map((bar) => bar.name)).toEqual(["generate_samples"]);
    const reward = rows[1][0];
    expect(reward.banded).toBe(true);
    expect(reward.count).toBe(32);
    expect(reward.average).toBeCloseTo(0.2);
    expect(reward.longest).toBe(0.3);
    // Concurrent scoring, so the time spent exceeds the span it covered.
    expect(reward.duration).toBe(6.4);
    expect(reward.end - reward.start).toBe(2.5);
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

    // The step, then the four runs it contains on one nested row beneath it.
    expect(rows).toHaveLength(2);
    expect(rows[0].map((bar) => [bar.name, bar.depth])).toEqual([["train_models", 0]]);
    expect(rows[1].map((bar) => bar.depth)).toEqual([1, 1, 1, 1]);
    expect(rows[1].map((bar) => bar.name)).toEqual([
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
    expect(rows[0].map((bar) => [bar.name, bar.start])).toEqual([
      ["train_models", 0],
      ["forward_backward", 2],
    ]);
  });

  it("has nothing to draw for a rollout that recorded no timing", () => {
    expect(rolloutTimeline({ roles: {} })).toEqual({ rows: [], span: 0 });
    expect(rolloutTimeline(null)).toEqual({ rows: [], span: 0 });
  });
});
