# Word-Puzzle Agent-Solver Base-Model Eval Runbook

This runbook records how to measure the **base model's** word-puzzle
solve rate through the standalone agent solver (`run_agent_solver.py`)
across all four thinking levels — `minimal`, `low`, `medium`, `high` —
and gives a table to fill in after each sweep.

The agent solver is eval-only: it builds a `GeminiApiSampler` plus a
mini-agent loop directly and runs each puzzle in a hardened Docker
sandboxed environment. It does NOT go through the training pipeline. "Base model"
means the publisher model (default `gemini-3.5-flash`) with **no**
`--endpoint-id` — i.e. no tuned checkpoint. To baseline a tuned
checkpoint instead, pass `--endpoint-id` (and set `--env` to the
region/tier the endpoint is deployed in) and note that in the table.

The only variable swept here is `--thinking-level`; every other flag
is held fixed so the four rows are comparable. Keep `--seed` constant
so all four levels see the identical puzzles.

## Prerequisites

Runs on any workstation or GCE VM; a running Docker daemon is required
either way.

1. **`uv` on `PATH`** (the repo's Python is managed by `uv`).
2. **Docker daemon running.**
3. **Pre-built sandbox image** `word-puzzle-agent-sandbox:latest`. The
   runner does NOT build it for you — build it once from this
   directory before the first run:

   ```bash
   docker build -t word-puzzle-agent-sandbox:latest \
       src/example_recipe/word_puzzle/agent_solver/minisweagent/
   ```

   Optionally verify the sandbox isolation is intact:

   ```bash
   src/example_recipe/word_puzzle/agent_solver/minisweagent/isolation_smoke_test.sh
   ```

4. **Application Default Credentials** with access to the Vertex
   `generateContent` surface on your `--project`:

   ```bash
   gcloud auth application-default login
   ```

## Run the four-level sweep

From the repo root. Replace `<your-gcp-project>` and pick `N`
(`--num-tasks`, default 10). The loop sweeps all four thinking levels,
holding the seed and every other flag fixed, and `tee`s each run's
stdout to its own log (the summary is stdout-only — there is no
aggregate output file, so capturing stdout is mandatory):

```bash
PROJECT=<your-gcp-project>
N=10
for LVL in minimal low medium high; do
  PYTHONPATH=src uv run python -m \
    example_recipe.word_puzzle.agent_solver.minisweagent.run_agent_solver \
    --num-tasks "$N" \
    --seed 42 \
    --project "$PROJECT" \
    --thinking-level "$LVL" \
    --trace-dir /tmp/agent_traces_"$LVL" \
    2>&1 | tee /tmp/agent_solver_"$LVL".log
done
```

Notes on the flags:

- `--seed 42` — deterministic puzzle selection; keep it identical
  across all four levels so every level scores the same puzzles.
- `--thinking-level` — the swept variable. `choices` are `minimal`,
  `low`, `medium`, `high`, and `""` (off / server default). `medium`
  is the runner default.
- Base model is the default `--model gemini-3.5-flash` with **no**
  `--endpoint-id`. The runner prints the resolved sampler URL to
  stderr before the first call — eyeball it to confirm you are hitting
  the publisher model and not a tuned endpoint.
- Per-tier split for `N=10` is 3 / 4 / 3 (tiers A / B / C); override
  with `--tier-a/-b/-c` if needed.
- Per-task trajectory JSONs land in each `--trace-dir`.

To run a single level instead of the sweep, run the inner command
once with your chosen `--thinking-level`.

## What to read from each run

Each run prints a summary block to stdout ending in an `OVERALL`
line. Read these off the tail of each `agent_solver_<level>.log`:

```
AGENT SOLVER SUMMARY -- model=<model>, thinking=<level>, n_tasks=<N>, wall=<s>
... per-task table ...
OVERALL: <n_correct>/<N> correct = <XX.X>%
  tier A: <ok>/<total> (<XX>%)
  tier B: <ok>/<total> (<XX>%)
  tier C: <ok>/<total> (<XX>%)
  avg model time / task: <XX.X>s
  avg bash calls / task: <XX.X>
  exit statuses: {<status>: <count>, ...}
```

- **OVERALL %** is the headline solve rate (`is_correct` = exact
  string match of submitted vs. expected answer).
- **tier A/B/C %** break the rate down by difficulty tier.
- **avg model time / task** and **avg bash calls / task** are the
  cost/effort signals.
- **exit statuses** is a histogram — anything other than clean
  submissions (e.g. limit/time-exceeded) shows up here and explains
  gaps in the solve rate.

## Results

Fill in one block per sweep. Record the run context once, then one
row per thinking level.

### Sweep: 2026-07-22 — model `gemini-3.5-flash`, N=5, seeds 42–46

- **Run by:** maintainer, by hand
- **Host:** local workstation (Docker local)
- **Project / env:** `<your-gcp-project>` / `prod-global`
- **Endpoint:** base publisher model (no `--endpoint-id`)
- **Agent settings:** `--step-limit 50`, `--wall-time-limit-seconds 600`,
  `--cost-limit 0.50`, `--max-generation-tokens 65536`,
  `--submit-retries 1`, per-action shell timeout 30s,
  `--max-consecutive-format-errors 2`, Docker sandbox (networkless,
  read-only, 512 MB RAM, 1 CPU, 64 PIDs)
- **Sampling:** each of the 5 tasks scored across 5 seeds (42, 43, 44,
  45, 46) — i.e. 5 independent samples per level. Values below are the
  **mean across the 5 seeds**; exit statuses are the **sum** over all
  25 task-runs per level.

| Thinking level | OVERALL % | tier A % | tier B % | tier C % | avg t_llm (s) | avg bash / task | exit statuses (sum / 25 runs) |
|---|---|---|---|---|---|---|---|
| minimal | 28.0 | 40.0 | 30.0 | 20.0 | 342.3 | 26.9 | Submitted 15, TimeExceeded 5, LimitsExceeded 3, Crash:ValueError 2 |
| low     | 36.0 | 60.0 | 40.0 | 20.0 | 263.7 | 22.2 | Submitted 22, TimeExceeded 2, LimitsExceeded 1 |
| medium  | 40.0 | 40.0 | 40.0 | 40.0 | 394.1 | 24.6 | Submitted 14, TimeExceeded 9, LimitsExceeded 2 |
| high    | 40.0 | 80.0 | 40.0 | 20.0 | 382.2 | 24.6 | Submitted 16, TimeExceeded 9 |

Observations:

- Solve rate rises with thinking budget: 28 → 36 → 40 → 40 %
  (minimal → low → medium → high); the gain flattens between medium
  and high on this tiny N=5 set.
- Per-seed OVERALL was stable (mostly 40 %; minimal and low dipped to
  20 % on a couple of seeds), so the ranking is consistent, not noise.
- `minimal` shows the only crashes (2 `Crash:ValueError`) and the most
  `LimitsExceeded` — with little thinking the agent more often derails
  or runs out of steps.
- `high` spends its budget on `TimeExceeded` (9/25) yet still tops tier
  A (80 %), suggesting the hardest tier benefits most from more
  thinking while easy tiers do not.

> Copy the block above for each new sweep; do not overwrite prior
> results — append.

### Sweep: 2026-07-23 — model `gemini-3.5-flash`, N=150 (full eval set), seeds 42–46

- **Run by:** maintainer, via the automated sweep script
- **Host:** dedicated GCE VM (us-central1-a, 16 vCPU, 62 GB)
- **Project / env:** `<your-gcp-project>` / `prod-global`
- **Endpoint:** base publisher model (no `--endpoint-id`)
- **Agent settings:** `--step-limit 50`, `--wall-time-limit-seconds 1800`,
  `--cost-limit 0.50`, `--max-generation-tokens 65536`,
  `--submit-retries 1`, per-action shell timeout 30s,
  `--max-consecutive-format-errors 2`, Docker sandbox (networkless,
  read-only, 512 MB RAM, 1 CPU, 64 PIDs)
- **Concurrency:** 64
- **Sampling:** all 150 eval tasks (50 per tier) scored across 5 seeds
  (42, 43, 44, 45, 46) — i.e. 5 independent samples per task per level.
  3,000 total task evaluations. Values below are the **mean across the
  5 seeds**; exit statuses are the **sum** over all 750 task-runs per
  level. Total wall time: ~8 hours.

#### Per-seed results

| Thinking level | seed 42 | seed 43 | seed 44 | seed 45 | seed 46 | **Mean** |
|---|---|---|---|---|---|---|
| minimal | 37.3% (56) | 42.7% (64) | 39.3% (59) | 40.0% (60) | 38.0% (57) | **39.5%** |
| low     | 46.7% (70) | 50.0% (75) | 46.7% (70) | 46.7% (70) | 46.0% (69) | **47.2%** |
| medium  | 48.7% (73) | 51.3% (77) | 49.3% (74) | 49.3% (74) | 43.3% (65) | **48.4%** |
| high    | 47.3% (71) | 46.0% (69) | 47.3% (71) | 50.7% (76) | 48.0% (72) | **47.9%** |

#### Mean summary (averaged over 5 seeds)

| Thinking level | OVERALL % | tier A % | tier B % | tier C % | avg t_llm (s) | avg bash / task | exit statuses (sum / 750 runs) |
|---|---|---|---|---|---|---|---|
| minimal | 39.5 | 50.8 | 45.6 | 22.0 | 294.5 | 24.1 | Submitted 580, LimitsExceeded 130, Crash:ValueError 16, Crash:RuntimeError 15, RepeatedFormatError 9 |
| low     | 47.2 | 57.2 | 56.4 | 28.0 | 277.6 | 21.0 | Submitted 677, LimitsExceeded 63, Crash:RuntimeError 7, Crash:ValueError 3 |
| medium  | 48.4 | 58.4 | 59.2 | 29.6 | 391.4 | 24.3 | Submitted 645, LimitsExceeded 93, Crash:RuntimeError 8, Crash:ValueError 4 |
| high    | 47.9 | 53.6 | 58.4 | 29.6 | 450.3 | 24.5 | Submitted 604, LimitsExceeded 102, Crash:RuntimeError 39, Crash:ValueError 5 |

#### Per-seed tier breakdowns

| Level | Seed | tier A | tier B | tier C |
|---|---|---|---|---|
| minimal | 42 | 46% | 42% | 24% |
| minimal | 43 | 56% | 50% | 22% |
| minimal | 44 | 52% | 44% | 22% |
| minimal | 45 | 52% | 46% | 22% |
| minimal | 46 | 48% | 46% | 20% |
| low | 42 | 62% | 52% | 26% |
| low | 43 | 60% | 62% | 28% |
| low | 44 | 56% | 58% | 26% |
| low | 45 | 54% | 56% | 30% |
| low | 46 | 54% | 54% | 30% |
| medium | 42 | 62% | 56% | 28% |
| medium | 43 | 62% | 64% | 28% |
| medium | 44 | 54% | 60% | 34% |
| medium | 45 | 60% | 60% | 28% |
| medium | 46 | 46% | 56% | 28% |
| high | 42 | 54% | 56% | 32% |
| high | 43 | 54% | 56% | 28% |
| high | 44 | 52% | 60% | 30% |
| high | 45 | 60% | 58% | 34% |
| high | 46 | 58% | 62% | 24% |

#### Observations

- **minimal → low is the dominant transition**: +7.7pp overall (39.5% →
  47.2%), with gains across all three tiers. This is where the thinking
  budget buys the most.
- **low → medium is marginal**: +1.2pp overall (47.2% → 48.4%). The gain
  concentrates in tier B (+2.8pp) and tier C (+1.6pp); tier A is flat.
- **high is NOT better than medium** — it's actually slightly worse
  (47.9% vs 48.4%). The extra thinking time burns more wall clock
  (450s vs 391s avg), causes more `Crash:RuntimeError` (39 vs 8 across
  750 runs), and slightly more `LimitsExceeded` (102 vs 93). The model
  "overthinks" on tasks where a shorter reasoning chain would have
  submitted correctly.
- **Tier C (hardest) saturates around 28-30%** regardless of thinking
  level beyond `low`. The difficulty ceiling is algorithmic, not
  reasoning-budget-limited.
- **`minimal` has the worst failure modes**: most `LimitsExceeded` (130),
  most `RepeatedFormatError` (9), and the most crashes overall. With
  little thinking the agent derails or exhausts its step budget more
  often.
- **Variance is low across seeds**: standard deviation of OVERALL % is
  ~2pp for each level, confirming the N=150 sample size is sufficient
  for stable estimates.
- **Compared to the N=5 pilot** (2026-07-22 sweep above): the full
  dataset confirms the same qualitative ordering (minimal < low ≈ medium
  ≈ high) but at higher absolute rates (39.5% vs 28% for minimal) —
  the pilot's tiny sample was biased toward harder tasks and had
  `wall_time_limit=600s` (now 1800s), which suppressed scores via
  `TimeExceeded`.

## Notes

- The summary is **stdout-only** — there is no machine-readable
  aggregate file. Always `tee` (or redirect) stdout, as the sweep loop
  above does.
- Keep `--seed`, `--num-tasks`, `--model`, and the tier split
  identical across the four levels; otherwise the rows are not
  comparable.
- Larger `N` tightens the estimate but costs more (real
  `generateContent` calls). Start with `N=10` for a quick read; bump
  to `N=50` for a firmer number.
- To baseline a **tuned checkpoint** rather than the base model, add
  `--endpoint-id <id>` and set `--env` to match where the endpoint is
  deployed; note it in the "Endpoint" field of the results block.
- The agent solver does not train; it is purely an eval harness for
  measuring solve rate. There is no weight update, no reward
  aggregation, and no config registry entry involved.
