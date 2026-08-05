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

describe("phaseSummaries", () => {
  it("aggregates totals across lanes and tracks rollout count", () => {
    const lanes = {
      roles: {
        driver: {
          role: "driver",
          rollout_id: 0,
          totals: {
            generate_rollouts: { total_duration_s: 1.2, max_duration_s: 0.6, count: 2 },
          },
        },
        rollout: {
          role: "rollout",
          rollout_id: 0,
          totals: {
            generate_rollouts: { total_duration_s: 0.8, max_duration_s: 0.4, count: 2 },
            custom_reward: { total_duration_s: 4.0, max_duration_s: 2.0, count: 8 },
          },
        },
      },
    };
    const rows = phaseSummaries(lanes);
    const gen = rows.find((r) => r.name === "generate_rollouts");
    expect(gen).toBeDefined();
    expect(gen.count).toBe(2);
    expect(gen.totalDuration).toBe(2.0);
    expect(gen.maxDuration).toBe(0.6);
    expect(gen.avgDuration).toBe(2.0 / 4);
    expect(gen.rolloutCount).toBe(1);
  });
});
