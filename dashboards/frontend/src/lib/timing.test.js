import { describe, expect, it } from "vitest";
import { colorFor, fmtSecs, labelFor, phaseSummaries, TIMING_COLORS, TIMING_LABELS } from "./timing.js";

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

const phase = (count, total, longest) => ({
  count,
  total_duration_s: total,
  longest_duration_s: longest,
  first_start_s: 0,
  last_end_s: total,
});

describe("phaseSummaries", () => {
  it("aggregates a phase across lanes and rollouts", () => {
    const rows = phaseSummaries({
      0: {
        roles: {
          driver: { role: "driver", phases: { generate_rollouts: phase(1, 1.2, 1.2) } },
          rollout: {
            role: "rollout",
            phases: { reward: phase(8, 4.0, 2.0) },
          },
        },
      },
      1: {
        roles: {
          driver: { role: "driver", phases: { generate_rollouts: phase(1, 0.8, 0.8) } },
        },
      },
    });

    const gen = rows.find((r) => r.name === "generate_rollouts");
    expect(gen.count).toBe(2);
    expect(gen.totalDuration).toBeCloseTo(2.0);
    expect(gen.longestDuration).toBe(1.2);
    expect(gen.avgDuration).toBeCloseTo(1.0);
    expect(gen.avgPerRollout).toBeCloseTo(1.0);
    expect(gen.rolloutsMeasured).toBe(2);
    expect(gen.rolloutCount).toBe(2);
  });

  it("says how many rollouts a phase is missing from", () => {
    const rows = phaseSummaries({
      0: { roles: { rollout: { role: "rollout", phases: { reward: phase(4, 2.0, 0.7) } } } },
      1: { roles: {} },
    });

    const reward = rows.find((r) => r.name === "reward");
    expect(reward.rolloutsMeasured).toBe(1);
    expect(reward.rolloutCount).toBe(2);
    // Averaged over the rollouts that recorded it, not the ones asked for.
    expect(reward.avgPerRollout).toBeCloseTo(2.0);
  });

  it("orders by time spent, longest first", () => {
    const rows = phaseSummaries({
      0: {
        roles: {
          driver: {
            role: "driver",
            phases: { weight_sync: phase(1, 0.5, 0.5), train_models: phase(1, 9.0, 9.0) },
          },
        },
      },
    });
    expect(rows.map((r) => r.name)).toEqual(["train_models", "weight_sync"]);
  });
});
