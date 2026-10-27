# Word Puzzle metrics

Metrics for the `word_puzzle` example recipe. For the metric
system as a whole — namespaces (`learning/` / `sampler/` / `pipeline/` +
`train/`/`eval/` twins), the `MetricSpec` registry, aggregation strategies, the
`emit_to_ml_platform` sink, and how to add a metric — see
`core/metric_spec.py` (registry + per-metric docs), `core/metric_aggregators.py`
(aggregators) and `core/sinks.py` (sinks).

## How this recipe emits metrics

Word puzzle uses the framework **`SingleStepAgent`** (`core.single_step_agent`)
and builds it **without** passing a custom `metric_aggregators` list
(`single_step/configs.py`, `src/train.py`, `src/eval.py`). So it runs
with `DEFAULT_METRIC_AGGREGATORS`
(`src/core/agent.py`) — the core aggregators only.

## Recipe-specific metrics

**None.** Word puzzle adds zero metrics on top of core. Its full metric surface
is exactly the core set registered in `core/metric_spec.py`:
`learning/*` (reward, sampling_accuracy, pass_at_k), `sampler/*`
(trajectory_status, finish_reason, generation_tokens_sum,
peak_context_tokens_mean, policy_step), `pipeline/*` (timing + bookkeeping), and
the `train/*` trainer wire + audit metrics — each with its `train/`/`eval/`
twin.

## Notes

- `stats.py` defines `WordPuzzleStepStats` (`generation_token_count`,
  `malformed_function_call`). This is consumed by the reward function only; it is
  **not** wired into any `MetricAggregator`, so it does not surface as a metric.
- `reward.py` / `reward_signals.py` compute rewards only — no metric emission.
- The `agent_solver/` subpackage is a standalone chat-completion agent-harness
  runner, separate from the RL training metric path; it does not contribute
  training/eval metrics.

If you add a word-puzzle-specific metric later, follow the pattern in
`core/metric_aggregators.py`: write a `MetricAggregator`,
register its `MetricSpec`(s), and pass
`DEFAULT_METRIC_AGGREGATORS + [YourAggregator()]` when constructing the agent.
