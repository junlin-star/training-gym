---
name: testing-dashboard
description: How to runtime-test the Training Gym observability dashboard (dashboards/app.py + Svelte SPA) — finding a deployed build, SPA URL patterns, timing APIs, and how to locate runs that exercise specific UI branches without redeploying.
---

# Testing the Training Gym dashboard

## Getting a running build

The dashboard is a Modal app (`dashboards/app.py`) that serves a pre-built Svelte SPA plus
JSON APIs. Prefer testing against an **already-deployed dev endpoint** for the branch, e.g.
`https://modal-labs-<user>-dev--training-gym-dashboard-fastapi-app.modal.run`. These are
public Modal web endpoints — no login needed.

Never run `modal deploy dashboards/app.py` while testing: that overwrites the shared main
dashboard. If you need your own build, deploy under a personal Modal environment/workspace,
or run the frontend locally (`cd dashboards/frontend && npm run dev`) proxying `/api` to a
deployed endpoint.

Confirm the deployment actually carries the branch's frontend before trusting a pass:
`curl -s <base>/ | grep -o 'assets/index-[A-Za-z0-9]*\.js'`, then grep that bundle for
strings the PR introduces (e.g. `curl -s <base>/assets/index-XXXX.js | grep -o "Substep timing across"`).

## URL patterns

- Runs list: `/training`
- Run detail: `/training/<run_id>?tab=summary` (also `tab=rollouts`, `tab=logs`)
- The Rollouts tab is a table; **click a row** to expand it — expanded rows are where
  per-rollout detail (timings, samples) renders.

## Useful APIs for cross-checking the UI

- `/api/runs` — all runs (hundreds; each entry has `run_id`, `status`)
- `/api/runs/<run_id>/rollouts` — the rollout ids the UI will list (ids can be sparse)
- `/api/runs/<run_id>/timings/<n>` — single rollout timing (`{"roles":{...}}`), fast and reliable
- `/api/runs/<run_id>/timings?rollout_ids=0,1,2,...` — batch, feeds the Summary phase table.
  **This one may be slow (up to ~20s) and may return HTTP 500 intermittently, more often
  with 3+ ids.** If a summary/aggregate block appears blank or flickers in the UI, probe this
  endpoint repeatedly from the shell before blaming the frontend:
  `for i in $(seq 1 8); do curl -s -o /dev/null -w "%{http_code} %{time_total}\n" "<url>"; done`

## Finding runs that exercise a specific UI branch

Rather than fault-injecting, scan the deployed metadata for real data that hits the branch.
Example — find a run where a *listed* rollout has empty timing (drives the
"no timing recorded for this rollout" state):

```python
# concurrent scan over /api/runs -> /rollouts -> /timings/<n>, print runs with empty roles
```

Useful classes of run to know about:
- Recent runs on a timing branch: full driver/actor/rollout lanes.
- Legacy runs (weeks old): render the old whole-run "Step & substep timeline" with zoom
  controls + Download JSON, and single-track per-step bars in expanded rollout rows. Always
  open one as a regression check.
- Cancelled/partial runs: often have a trailing rollout with no timing — good for empty states.

## Gotchas

- The run detail page polls; a block that depends on a flaky endpoint can appear and then
  vanish between polls. Take screenshots of both states rather than assuming a single sample
  is representative.
- The Summary tab's timing table can take minutes to appear on a slow/failing batch endpoint;
  wait at least 2–3 minutes before declaring it missing.
- Every timing number rendered in the UI has an exact API counterpart
  (`count` / `total_duration_s` / `longest_duration_s`) — verify `average == total/count`
  for phases with `count > 1`, which the UI labels "… over N runs".
- Element `title` attributes on lane rows contain the full "X total, ran N×, longest …,
  first_start → last_end" text; reading the stripped DOM is often faster than zooming.

## Devin Secrets Needed

None — the dev dashboard endpoints are public. Modal CLI auth is only needed if you deploy,
which you should avoid for the shared app.
