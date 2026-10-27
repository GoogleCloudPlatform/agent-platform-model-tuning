# Agent Platform Model Tuning

A client library and example recipes for multi-step reinforcement learning
(MSRL) fine-tuning on Vertex AI. The client drives rollouts against a tuning
job, collects trajectories in the open
[ATIF](https://github.com/laude-institute/terminal-bench) trajectory format,
computes rewards, and submits training steps.

One example recipe is included:

- **`word_puzzle`** — a task where the model recovers an original word from a
  transformed form. It ships in both a single-step form and a multi-step,
  tool-using agent form, and is a good first end-to-end run.

## Quickstart

Setup (uv): a fresh clone runs on any machine with [`uv`](https://docs.astral.sh/uv/)
installed — no manual venv. `uv run` auto-creates `.venv/` from the lockfile on
first use. Python is pinned to 3.12 via `.python-version`.

Dependencies are resolved from [PyPI](https://pypi.org) and pinned in the
committed `uv.lock`.

See [Running an experiment](#running-an-experiment) for the launcher and the
recipe `config_name`s.

For direct `uv` use:
```sh
uv run --extra test python -m pytest
```

### Running an experiment

There is **one recipe-agnostic entry point** — you pick the recipe by
`config_name`, and the recipe hyperparameters (train schedule, reward
shaping, backend, etc.) are **baked into the checked-in config**, not passed
on the CLI. Only launch-env fields (`vertex_*`, `log_dir`, `resume_*`,
`train_data_path` / `eval_data_path`) are overridable as bare args.

| Recipe | `config_name` |
|--------|---------------|
| word_puzzle async (single-step) | `word_puzzle_async` |
| word_puzzle sync | `word_puzzle_sync` |

**Use the launcher script** (recommended) — it sets `PYTHONPATH`, keeps the
run alive after your SSH session ends (session-survival), and presets the
`VERTEX_*` defaults:

```sh
# Word puzzle, async, against the Vertex MSRL API:
bash scripts/word_puzzle_async_train_customer.sh

# Reuse an existing RUNNING tuning job instead of creating one:
VERTEX_EXISTING_JOB_ID=<job_id> \
  bash scripts/word_puzzle_async_train_customer.sh

# Run detached so it survives your shell:
tmux new -d -s msrl \
  'bash scripts/word_puzzle_async_train_customer.sh 2>&1 | tee launch.log; bash'
```

All launchers honour `VERTEX_PROJECT`, `VERTEX_BASE_MODEL`, `VERTEX_ENV`
and `VERTEX_EXISTING_JOB_ID`.

**Direct invocation** (what the script wraps):

```sh
PYTHONPATH=src uv run python src/train.py word_puzzle_async \
    vertex_env=prod \
    vertex_project=<your-project-number> \
    vertex_base_model=<base-model> \
    log_dir=/tmp/wp_async_run
```

Eval-only against an existing RUNNING tuning job (same `config_name` surface):

```sh
PYTHONPATH=src uv run python src/eval.py word_puzzle_async \
    vertex_existing_job_id=<id>
```

Create a Vertex MSRL tuning job (waits ~30 min for `JOB_STATE_RUNNING`,
prints the numeric job id as the last line of stdout):

```sh
PYTHONPATH=src uv run python -m tools.create_tuning_job
```

Run tests:

```sh
uv run --extra test python -m pytest
```

## Architecture

Concept layers compose orthogonally:

```
Task (Env + Dataset) × Backend (Sampler + Trainer) × Loop (sync/async)
                     × Entry point (train/<name>.py)
```

Agents bridge a Sampler to an Env. `SingleStepAgent` covers any
single-prompt task. The training loops are backend-agnostic — they depend
only on the protocols in `core/` (`Agent`, `Sampler`, `Trainer`), so
any backend that implements them plugs in without touching loop code.

Call hierarchy:

```
src/train.py <config_name>              entry: load composed RunConfig,
                                        apply launch overrides, run_training
    example_recipe/run_config_entry.py  builds components via config.make() seam
    pipeline/train_and_eval.py
        backend-agnostic train+eval loop; depends only on core protocols
    backends/vertex_msrl.py
        Sampler + Trainer for Vertex MSRL
    core/
        Agent / Sampler / Trainer protocols, JsonlDataset, BatchIterator,
        SingleStepAgent, TrajectoryGroup
    word_puzzle/
        Task: Env + Agent + reward utils
```

`SingleStepAgent` (in `core/`) wraps any single-prompt env;
`MultiStepAgent` is the multi-step counterpart that implements tool-calling
loops, prompt-too-long handling, and per-step reward attribution.

## Repository layout

```
core/                           Core protocols + generic Dataset/Iterator/Agent
word_puzzle/                    Word puzzle env, JSONL parser, eval helper
backends/                       Vertex MSRL sampler/trainer + factory
src/train.py, src/eval.py       Entry points (positional config name)
pipeline/train_and_eval.py      Backend-agnostic train+eval loop (library)
reward_normalizer.py            Reward normalization (GRPO-style)
ml_log.py, logging_utils.py     Metric logging utilities
rl_metrics.py                   KL / IS-weight metrics
scripts/                        Shell wrappers around the entry points.
src/tools/                       `create_tuning_job.py` (one-shot Vertex
                                tuning-job creator); run via
                                `python -m tools.create_tuning_job`.
tools/                          Experiment plotting / metric merging utilities
tests/                          Unit tests (see tests/README.md)
```

For the authoritative, up-to-date top-two-layer tree of `src/`, see the source tree.

## Adding a new task

Implement an `Env` subclass (see `core.Env` and
`word_puzzle.WordPuzzleEnv` for the simplest example). If your data needs
reshaping, pass a `transform_fn` to `JsonlDataset`. Single-step tasks can
reuse `SingleStepAgent`; multi-step tasks subclass `Agent`.

## Adding a new backend

Implement two classes against the protocols in `core/`:

  - `Sampler.sample(prompt, *, num_samples=1, ...) -> list[dict]` with at
    least `completion_text` (full schema in `SingleStepAgent`'s docstring).
  - `Trainer.train_step(trajectories) -> dict[str, float]` of metrics.

Place them under `backends/<your_backend>.py` and extend
`backends.factory.build_backend` to dispatch on a new
`config.backend` value. Loop libraries do not need to change.

## Testing

See `tests/README.md` for layout, fixtures, and conventions.

```sh
uv run --extra test python -m pytest                          # all (~2s)
uv run --extra test python -m pytest tests/unit -v
uv run --extra test python -m pytest --cov --cov-report=html  # coverage -> htmlcov/
```

## Vertex AI Experiments logger (optional)

Training runs can additionally stream their hparams and per-step
metrics to a [Vertex AI
Experiment](https://cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments)
alongside the existing `metrics.jsonl` and console output. This is
opt-in — nothing changes for callers who don't set `vertex_experiment`.

Install the extra:

    uv sync --extra vertex_logger

Ensure ADC is set up (once per workstation):

    gcloud auth application-default login

Vertex AI Experiments logging is driven by the checked-in config's
`vertex_experiment` task field (no longer a CLI knob); when it is set, any
launcher streams hparams + metrics to that experiment. Install the extra and
run a recipe as usual:

    uv run --extra vertex_logger \
      bash scripts/word_puzzle_async_train_customer.sh

The run appears in Cloud Console → Vertex AI → Experiments → *<experiment>*.
Its Parameters column shows the flattened `Config`, and its Metrics tab
shows per-step training/eval metrics.

Optional knobs (all default to sensible values):

| Config field | Meaning | Default |
|---|---|---|
| `vertex_experiment` | Experiment name; empty = disable Vertex logging | `""` |
| `vertex_project` | GCP project number | (none — set explicitly) |
| `vertex_location` | GCP region | `us-central1` |
| `vertex_experiment_tensorboard` | Backing TB. `""` = auto-lookup-or-create `vmsrl-default`; `"disable"` = skip TB (final-snapshot metrics still work, per-step curves won't); full `projects/.../tensorboards/...` = use as-is | `""` |
| `vertex_experiment_run_name` | Run name within the experiment; empty = derive from `basename(log_dir_subdir)` (the timestamp) | `""` |

**Cost note:** the auto-provisioned TensorBoard instance costs ~$300/mo
per region and stays around after runs finish. If you already have a
shared TB instance, pass its resource name to
`vertex_experiment_tensorboard` to reuse it. Pass `"disable"` if you
only care about the Experiment Run page and don't need classic
TensorBoard curves.

Vertex API failures during a run (network blip, quota, etc.) are
logged as warnings but never crash the training run — the JSONL logger
remains the source of truth.

## Weights & Biases logger (optional)

Training runs can also stream hparams and per-step metrics to a
[Weights & Biases](https://wandb.ai) run, alongside `metrics.jsonl`,
the console, and (optionally) Vertex. Opt-in — nothing changes for
callers who don't set `wandb_project`.

Install the extra:

    uv sync --extra wandb_logger

**Offline by default.** Unless you pass `wandb_mode=online`, the logger
sets `WANDB_MODE=offline` and writes a local `wandb/` run dir under the
run's log dir — no network, so it works on egress-restricted training
VMs. Publish it afterwards from a machine with access:

    wandb login            # once, key from https://wandb.ai/authorize
    wandb sync <log_dir>/wandb/latest-run

For online logging (dev boxes, or a VM with egress / an internal
server), authenticate first, then enable it:

    wandb login            # or: export WANDB_API_KEY=...
    # pass wandb_project and wandb_mode to your training entry point

Point at an internal W&B server with `export WANDB_BASE_URL=<url>`.

Optional knobs:

| Config field | Meaning | Default |
|---|---|---|
| `wandb_project` | W&B project; empty = disable wandb logging | `""` |
| `wandb_entity` | W&B entity (team/user); empty = wandb default | `""` |
| `wandb_run_name` | Run name; empty = `basename(log_dir)` | `""` |
| `wandb_mode` | `offline` (default, local, no network) or `online` | `offline` |

wandb API failures during a run are logged as warnings but never crash
the training run — the JSONL logger remains the source of truth.

## Project status

Research / experimental client. Public APIs are not stable; expect breaking
changes between commits.

## Security

Eligibility for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security) is determined by the [Google Open Source Software Vulnerability Reward Program Rules](https://bughunters.google.com/about/rules/open-source/google-open-source-software-vulnerability-reward-program-rules).
