<script>
  import { fmtSecs, labelFor, phaseSummaries } from "../lib/timing.js";

  let { lanes = null } = $props();

  let rows = $derived(phaseSummaries(lanes || {}));
</script>

<div class="phase-summary">
  {#if rows.length === 0}
    <div class="empty">No phase totals available.</div>
  {:else}
    <table>
      <thead>
        <tr>
          <th>Phase</th>
          <th>Rollouts</th>
          <th>Total</th>
          <th>Average</th>
          <th>Max</th>
        </tr>
      </thead>
      <tbody>
        {#each rows as row (row.name)}
          <tr>
            <td>{labelFor(row.name)}</td>
            <td>
              {row.count}/{row.rolloutCount}
              {#if row.count < row.rolloutCount}
                <span
                  class="missing"
                  title="Not measured on every rollout shown: the phase did not run, or its record never arrived."
                >
                  · missing in {row.rolloutCount - row.count}
                </span>
              {/if}
            </td>
            <td>{fmtSecs(row.totalDuration)}</td>
            <td>{fmtSecs(row.avgDuration)}</td>
            <td>{fmtSecs(row.maxDuration)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}
</div>

<style>
  .phase-summary {
    font-size: 0.85rem;
  }
  table {
    width: 100%;
    border-collapse: collapse;
  }
  th,
  td {
    padding: 0.25rem 0.5rem;
    text-align: left;
  }
  th {
    color: var(--color-c-gray-45, #6e6e6e);
    font-weight: 500;
  }
  .missing {
    color: var(--color-c-gray-45, #6e6e6e);
    margin-left: 0.25rem;
  }
  .empty {
    color: var(--color-c-gray-45, #6e6e6e);
    padding: 0.5rem 0;
  }
</style>
