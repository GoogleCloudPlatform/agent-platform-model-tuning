# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logging utilities"""

import json
import logging
import os
import re
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from pathlib import Path
import shutil

from rich.console import Console
from rich.table import Table

from core.sinks import (
    ConsoleSink,
    JsonlSink,
    MLPlatformSink,
    Sink,
    SinkDispatcher,
    TraceLogSink,
    TraceLogWriter,
)
from utils.rollout_log import install_rollout_log_handler

logger = logging.getLogger(__name__)


def _dump_config(config: Any) -> Any:
    """Convert configuration object to JSON-serializable format."""
    if hasattr(config, "to_dict"):
        return config.to_dict()
    elif isinstance(config, dict):
        return {k: _dump_config(v) for k, v in config.items()}
    elif isinstance(config, (list, tuple)):
        return [_dump_config(item) for item in config]
    elif hasattr(config, "__dict__"):
        return {
            k: _dump_config(v)
            for k, v in config.__dict__.items()
            if not k.startswith(("_", "X_"))
        }
    elif callable(config):
        return f"{config.__module__}.{config.__name__}"
    else:
        return config


def _flatten_hparams(
    d: Dict[str, Any], sep: str = ".", _prefix: str = "", _max_len: int = 1000
) -> Dict[str, Any]:
    """Flatten a (possibly nested) config dict into a form suitable for
    Vertex AI's aiplatform.log_params, which accepts Dict[str, float | int | str].

    Nested dicts are dot-flattened. Lists/tuples/None/arbitrary objects are
    stringified. Strings longer than ``_max_len`` are truncated with a
    trailing ellipsis so operators can see the value was cut.
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{_prefix}{sep}{k}" if _prefix else k
        if isinstance(v, dict):
            out.update(_flatten_hparams(v, sep=sep, _prefix=key, _max_len=_max_len))
        elif isinstance(v, (list, tuple)):
            out[key] = _truncate(repr(v), _max_len)
        elif v is None:
            out[key] = "None"
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = v
        elif isinstance(v, bool):
            out[key] = v
        elif isinstance(v, str):
            out[key] = _truncate(v, _max_len)
        else:
            out[key] = _truncate(str(v), _max_len)
    return out


def _truncate(s: str, max_len: int) -> str:
    """Truncate ``s`` to ``max_len`` chars, appending an ellipsis if cut."""
    if len(s) <= max_len:
        return s
    return s[:max_len] + "…"


class Logger(ABC):
    """Abstract base class for loggers."""

    @abstractmethod
    def log_hparams(self, config: Any) -> None:
        """Log hyperparameters/configuration."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        """Log metrics dictionary with optional step number."""
        pass

    def close(self) -> None:
        """Cleanup when done (optional to implement)."""
        pass


class _PermissiveJSONEncoder(json.JSONEncoder):
    """JSON encoder that stringifies non-encodable objects to their type name."""

    def default(self, o: Any) -> Any:
        try:
            return super().default(o)
        except TypeError:
            return str(type(o))


class JsonLogger(Logger):
    """Logger that writes metrics to a JSONL file."""

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self._logged_hparams = False

    def log_hparams(self, config: Any) -> None:
        """Log hyperparameters to a separate config.json file."""
        if not self._logged_hparams:
            config_dict = _dump_config(config)
            config_file = self.log_dir / "config.json"
            with open(config_file, "w") as f:
                json.dump(config_dict, f, indent=2, cls=_PermissiveJSONEncoder)
            self._logged_hparams = True

    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        """Append metrics to JSONL file."""
        log_entry = {"step": step} if step is not None else {}
        log_entry.update(metrics)

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
            logger.info("Wrote metrics to %s", self.metrics_file)


class PrettyPrintLogger(Logger):
    """Logger that displays metrics in a formatted table in the console."""

    def __init__(self):
        self.console = Console()

    def log_hparams(self, config: Any) -> None:
        """Print configuration summary."""
        config_dict = _dump_config(config)
        with _rich_console_use_logger(self.console):
            self.console.print("[bold cyan]Configuration:[/bold cyan]")
            for key, value in config_dict.items():
                self.console.print(f"  {key}: {_maybe_truncate_repr(value)}")

    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        """Display metrics in console."""
        if not metrics:
            return

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        if step is not None:
            table.title = f"Step {step}"

        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                value_str = f"{value:.6f}"
            else:
                value_str = str(value)
            table.add_row(key, value_str)

        with _rich_console_use_logger(self.console):
            self.console.print(table)


def _maybe_truncate_repr(value: Any) -> str:
    repr_value = repr(value)
    if len(repr_value) > 256:
        return repr_value[:128] + " ... " + repr_value[-128:]
    return repr_value


@contextmanager
def _rich_console_use_logger(console: Console):
    with console.capture() as capture:
        yield
    # Leading newline keeps table formatting readable when the logger
    # prepends its own prefix line.
    logger.info("\n" + capture.get().rstrip())


class VertexExperimentLogger(Logger):
    """Logger that streams hparams and per-step metrics to Vertex AI Experiments.

    Wrapped by ``core.sinks.MLPlatformSink`` in the sink dispatch layer.
    Requires the ``vertex_logger`` extra for ``google-cloud-aiplatform``.

    Runtime API failures are logged as warnings but never re-raised so
    losing the Vertex upload can't crash a training run. Startup
    failures (import, auth, missing project, TB creation) DO raise.
    """

    def __init__(
        self,
        *,
        experiment: str,
        run_name: str,
        project: str,
        location: str,
        tensorboard: Any = None,
        resume: bool = True,
    ):
        # Lazy import keeps the base install free of the ~150MB aiplatform dep.
        try:
            from google.cloud import aiplatform  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "VertexExperimentLogger requires google-cloud-aiplatform. "
                "Install with: uv sync --extra vertex_logger"
            ) from e
        self._aiplatform = aiplatform
        self._experiment = experiment
        self._run_name = run_name
        self._project = project
        self._location = location
        self._tensorboard = tensorboard
        self._resume = resume
        self._hparams_logged = False
        self._failure_count = 0
        experiment_tensorboard = self._resolve_tensorboard(tensorboard)

        self._aiplatform.init(
            project=project,
            location=location,
            experiment=experiment,
            experiment_tensorboard=experiment_tensorboard,
        )
        # aiplatform.start_run(name, resume=True) requires the run to already
        # exist (raises NotFound otherwise). To get "attach if exists, else
        # create" so restarts re-attach cleanly, try resume=True first and
        # fall back to a fresh create on NotFound.
        try:
            self._aiplatform.start_run(run_name, resume=resume)
        except Exception as e:  # noqa: BLE001
            not_found = (
                "not found" in str(e).lower() or "notfound" in type(e).__name__.lower()
            )
            if resume and not_found:
                logger.info(
                    "VertexExperimentLogger: run %r not found in experiment "
                    "%r — creating fresh (resume=True was best-effort).",
                    run_name,
                    experiment,
                )
                self._aiplatform.start_run(run_name, resume=False)
            else:
                raise

    _DEFAULT_TB_DISPLAY_NAME = "gemini-rl-exp"

    def _resolve_tensorboard(self, tensorboard: Any) -> Any:
        """Return the value to pass as ``experiment_tensorboard`` to init.

        Accepts ``False`` (disable), a full ``projects/...`` resource
        name string, or ``None`` (auto-lookup-or-create the default TB).
        """
        if tensorboard is False:
            return False
        if isinstance(tensorboard, str):
            if tensorboard.startswith("projects/"):
                return tensorboard
            raise ValueError(
                f"vertex_tensorboard string must be a full resource name "
                f"starting with 'projects/'; got {tensorboard!r}. Pass False "
                f"to disable, None to auto-provision "
                f"'{self._DEFAULT_TB_DISPLAY_NAME}', or a full resource name."
            )
        if tensorboard is None:
            # project/location must be explicit: this runs before
            # aiplatform.init(), so the ambient default would otherwise be the
            # caller's own project. Picking up a TensorBoard from a different
            # project than the experiment makes init() raise.
            existing = list(
                self._aiplatform.Tensorboard.list(
                    filter=f'display_name="{self._DEFAULT_TB_DISPLAY_NAME}"',
                    project=self._project,
                    location=self._location,
                )
            )
            if existing:
                resource = existing[0].resource_name
                logger.info(
                    "VertexExperimentLogger: reusing existing default TensorBoard %s",
                    resource,
                )
                return resource
            logger.info(
                "VertexExperimentLogger: no TensorBoard named %r found in "
                "%s/%s — auto-provisioning one (this incurs ongoing cost, "
                "~$300/mo per instance in most regions).",
                self._DEFAULT_TB_DISPLAY_NAME,
                self._project,
                self._location,
            )
            created = self._aiplatform.Tensorboard.create(
                display_name=self._DEFAULT_TB_DISPLAY_NAME,
                project=self._project,
                location=self._location,
            )
            logger.info(
                "VertexExperimentLogger: created TensorBoard %s",
                created.resource_name,
            )
            return created.resource_name
        raise ValueError(
            f"vertex_tensorboard must be False (disable), None (auto-provision "
            f"'{self._DEFAULT_TB_DISPLAY_NAME}'), or a full resource name string; got "
            f"{type(tensorboard).__name__} {tensorboard!r}."
        )

    def log_hparams(self, config: Any) -> None:
        if self._hparams_logged:
            return
        try:
            config_dict = _dump_config(config)
            if not isinstance(config_dict, dict):
                # _dump_config may return a scalar; wrap it so
                # log_params gets a dict.
                config_dict = {"config": config_dict}
            flat = _flatten_hparams(config_dict)
            self._aiplatform.log_params(flat)
        except Exception as e:  # noqa: BLE001
            self._record_failure("log_hparams", e)
        finally:
            # Set even on failure so we don't retry-storm a persistent 4xx.
            self._hparams_logged = True

    def _record_failure(self, op: str, exc: Exception) -> None:
        """Log a Vertex API failure; throttle traceback to first occurrence."""
        self._failure_count += 1
        if self._failure_count == 1:
            logger.warning(
                "VertexExperimentLogger.%s failed: %s (first occurrence — "
                "training continues; subsequent failures will be logged as "
                "one-liners without tracebacks)",
                op,
                exc,
                exc_info=True,
            )
        else:
            logger.warning(
                "VertexExperimentLogger.%s failed: %s (occurrence %d)",
                op,
                exc,
                self._failure_count,
            )

    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        numeric: Dict[str, float] = {}
        for k, v in metrics.items():
            # bool is a subclass of int; drop it before the int/float branch.
            if isinstance(v, bool):
                logger.debug(
                    "VertexExperimentLogger: dropping non-numeric metric %r=%r",
                    k,
                    v,
                )
                continue
            if isinstance(v, (int, float)):
                numeric[k] = float(v)
            else:
                logger.debug(
                    "VertexExperimentLogger: dropping non-numeric metric %r=%r",
                    k,
                    v,
                )
        if not numeric:
            return
        try:
            if step is not None:
                self._aiplatform.log_time_series_metrics(numeric, step=step)
            else:
                self._aiplatform.log_metrics(numeric)
        except Exception as e:  # noqa: BLE001
            self._record_failure("log_metrics", e)

    # ``end_run()`` issues a synchronous, deadline-less gRPC UpdateContext
    # to the metadata service. When that call wedges it blocks the main
    # thread inside the teardown ``finally``, so SIGTERM never completes
    # and the run's terminal status is never written. Bound it.
    _CLOSE_TIMEOUT_SECONDS = 60.0

    def close(self) -> None:
        def _end_run() -> None:
            try:
                self._aiplatform.end_run()
            except Exception as e:  # noqa: BLE001
                self._record_failure("close", e)

        # Daemon thread: if end_run() never returns, the blocked call is
        # abandoned rather than holding up interpreter exit.
        worker = threading.Thread(
            target=_end_run, name="vertex-experiment-close", daemon=True
        )
        worker.start()
        worker.join(self._CLOSE_TIMEOUT_SECONDS)
        if worker.is_alive():
            logger.warning(
                "VertexExperimentLogger.close: end_run() did not return within "
                "%.0fs — abandoning it and continuing teardown. The Vertex "
                "Experiment run %r may stay in a RUNNING state.",
                self._CLOSE_TIMEOUT_SECONDS,
                self._run_name,
            )


def resolve_vertex_tensorboard_config(tb_config: str) -> Any:
    """Map an entry-point Config string to VertexExperimentLogger's sentinel.

    CLI Configs are dataclass string fields, so ``False``/``None`` can't
    be passed directly; use the string convention:
      - ``""``            → ``None`` (auto-provision)
      - ``"disable"``     → ``False`` (disable TB)
      - ``"projects/..."``→ passthrough (that specific TensorBoard)

    Raises:
      TypeError: If ``tb_config`` is not a ``str``.
      ValueError: If ``tb_config`` is a string but doesn't match one of
        the three accepted forms.
    """
    if not isinstance(tb_config, str):
        raise TypeError(
            f"tb_config must be a str; got {type(tb_config).__name__} {tb_config!r}"
        )
    if tb_config == "":
        return None
    if tb_config == "disable":
        return False
    if tb_config.startswith("projects/"):
        return tb_config
    raise ValueError(
        f"vertex_tensorboard config string must be '' (auto-provision), "
        f"'disable' (off), or a full resource name starting with "
        f"'projects/'; got {tb_config!r}."
    )


# Vertex ExperimentRun names must match ^[a-z0-9][a-z0-9-]{0,127}$
# (lowercase alphanumerics + hyphens, <=128 chars). See
# https://cloud.google.com/vertex-ai/docs/experiments/create-experiment
_VERTEX_RUN_NAME_MAX = 128


def _sanitize_vertex_run_name(name: str) -> str:
    """Coerce an arbitrary string into a valid Vertex ExperimentRun name.

    Lowercases, replaces any run of disallowed chars with a single hyphen,
    strips leading/trailing hyphens, ensures a leading alphanumeric, and
    truncates to 128 chars. Empty result falls back to ``"run"``.
    """
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not s or not s[0].isalnum():
        s = "r-" + s if s else "run"
    return s[:_VERTEX_RUN_NAME_MAX].rstrip("-") or "run"


def compose_vertex_run_name(job_id: str, run_name: str | None) -> str:
    """Prepend the tuning ``job_id`` to a run name so a Vertex ExperimentRun
    is traceable to its tuning job at a glance.

    Returns a sanitized ``"{job_id}-{run_name}"`` (e.g.
    ``"1234567890123456789-my-vm-20260101-000000"``). When ``run_name`` is
    empty/None, returns just the sanitized ``job_id`` (still traceable). The
    result is coerced to Vertex's ``^[a-z0-9][a-z0-9-]{0,127}$`` constraint.
    """
    base = f"{job_id}-{run_name}" if run_name else str(job_id)
    return _sanitize_vertex_run_name(base)


def default_vertex_run_name(log_dir_path: Path) -> str:
    """Derive a Vertex run name that embeds the tuning job id.

    All recipes build their log dir as ``{log_dir}/{job_id}/{timestamp}``
    (see each recipe's ``_get_log_subdir``), so the job id is the parent
    directory name and the timestamp is the leaf. Compose them into
    ``job-{job_id}-{timestamp}`` so a Vertex run is traceable back to its
    tuning job at a glance (the SDK's own default is just
    ``{hostname}-{timestamp}``, which hides the job id).

    Falls back to just the leaf name when the parent isn't a plausible
    numeric job id (e.g. a flat log dir), so this never regresses a
    non-standard layout.
    """
    leaf = log_dir_path.name
    parent = log_dir_path.parent.name
    if parent and parent.isdigit():
        return _sanitize_vertex_run_name(f"job-{parent}-{leaf}")
    return _sanitize_vertex_run_name(leaf)


class WandbLogger(Logger):
    """Logger that streams hparams and per-step metrics to Weights & Biases.

    Opt-in backend for ``MLPlatformSink``. Requires the ``wandb_logger``
    extra (``uv sync --extra wandb_logger``).

    Offline by default: unless ``mode="online"`` is requested, ``WANDB_MODE``
    is set to ``offline`` before ``wandb.init`` so a training VM with blocked
    egress writes a local ``wandb/`` run dir and never blocks on the network
    (``wandb sync`` it later, or point ``WANDB_BASE_URL`` at an internal
    server and pass ``mode="online"``).

    Failure model mirrors VertexExperimentLogger: startup failures (import,
    init) RAISE so misconfig surfaces immediately; per-step runtime failures
    are logged via ``_record_failure`` and NEVER re-raised.
    """

    def __init__(
        self,
        *,
        project: str,
        run_name: str,
        log_dir: str,
        entity: str = "",
        mode: str = "offline",
        resume: str = "allow",
    ):
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "WandbLogger requires wandb. Install with: uv sync --extra wandb_logger"
            ) from e
        self._wandb = wandb
        self._failure_count = 0
        self._hparams_logged = False
        if mode != "online":
            os.environ["WANDB_MODE"] = mode
        self._run = wandb.init(
            project=project,
            name=run_name,
            entity=entity or None,
            mode=mode,
            resume=resume,
            dir=log_dir,
        )

    def _record_failure(self, op: str, exc: Exception) -> None:
        """Log a wandb API failure; throttle traceback to first occurrence."""
        self._failure_count += 1
        if self._failure_count == 1:
            logger.warning(
                "WandbLogger.%s failed: %s (first occurrence -- training "
                "continues; subsequent failures logged as one-liners)",
                op,
                exc,
                exc_info=True,
            )
        else:
            logger.warning(
                "WandbLogger.%s failed: %s (occurrence %d)",
                op,
                exc,
                self._failure_count,
            )

    def log_hparams(self, config: Any) -> None:
        if self._hparams_logged:
            return
        try:
            config_dict = _dump_config(config)
            if not isinstance(config_dict, dict):
                config_dict = {"config": config_dict}
            self._run.config.update(_flatten_hparams(config_dict))
        except Exception as e:  # noqa: BLE001
            self._record_failure("log_hparams", e)
        finally:
            self._hparams_logged = True

    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        numeric: Dict[str, float] = {}
        for k, v in metrics.items():
            if isinstance(v, bool):
                logger.debug("WandbLogger: dropping non-numeric metric %r=%r", k, v)
                continue
            if isinstance(v, (int, float)):
                numeric[k] = float(v)
            else:
                logger.debug("WandbLogger: dropping non-numeric metric %r=%r", k, v)
        if not numeric:
            return
        try:
            if step is not None:
                self._run.log(numeric, step=step)
            else:
                self._run.log(numeric)
        except Exception as e:  # noqa: BLE001
            self._record_failure("log_metrics", e)

    def close(self) -> None:
        try:
            self._run.finish()
        except Exception as e:  # noqa: BLE001
            self._record_failure("close", e)


def setup_logging(
    log_dir: str,
    config: Any | None = None,
    do_configure_logging_module: bool = True,
    clear_log_dir: bool = False,
    *,
    vertex_experiment: str | None = None,
    vertex_project: str | None = None,
    vertex_location: str = "us-central1",
    vertex_tensorboard: Any = None,
    vertex_run_name: str | None = None,
    wandb_project: str | None = None,
    wandb_entity: str = "",
    wandb_run_name: str | None = None,
    wandb_mode: str = "offline",
    extra_ml_platform_backends: list[Any] | None = None,
) -> SinkDispatcher:
    """Set up logging infrastructure with JSON + Console backends,
    optionally plus a Vertex AI Experiments backend.

    Args:
        log_dir: Directory for logs.
        config: Configuration object to log via ``log_hparams``.
        do_configure_logging_module: Whether to configure stdlib logging.
        clear_log_dir: Wipe ``log_dir`` before setting up.
        vertex_experiment: If set, also stream metrics/hparams to a Vertex
            AI Experiments run under this name. Requires the
            ``vertex_logger`` extra.
        vertex_project: GCP project for the Vertex Experiment. Required
            when ``vertex_experiment`` is set.
        vertex_location: GCP region for the Vertex Experiment.
        vertex_tensorboard: Backing TensorBoard: ``False`` disables,
            ``None`` auto-lookup-or-creates a default TB, or a full
            ``projects/.../tensorboards/...`` resource name.
        vertex_run_name: Run name within the experiment. Defaults to
            ``job-{job_id}-{timestamp}`` derived from ``log_dir`` (via
            ``default_vertex_run_name``) so the run is traceable back to
            its tuning job.
        wandb_project: If set, also stream metrics/hparams to a Weights &
            Biases run under this project. When None (default) no wandb
            code runs. Requires the ``wandb_logger`` extra.
        wandb_entity: W&B entity (team/user); empty uses the wandb default.
        wandb_run_name: Run name within the project. Defaults to
            ``basename(log_dir)`` if not provided.
        wandb_mode: W&B mode; ``"offline"`` (default) writes a local run
            dir; ``"online"`` requires egress or ``WANDB_BASE_URL``.
        extra_ml_platform_backends: Additional backend loggers to wrap in
            ``MLPlatformSink`` and attach to the dispatcher.

    Returns:
        SinkDispatcher wrapping 3 to 5 sinks (jsonl / console / trace,
        plus ML-platform sinks for ``vertex_experiment``,
        ``wandb_project``, and any ``extra_ml_platform_backends``).
    """
    log_dir_path = Path(log_dir).expanduser()
    if clear_log_dir and log_dir_path.exists():
        shutil.rmtree(log_dir_path)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    json_backend = JsonLogger(log_dir_path)
    console_backend = PrettyPrintLogger()
    trace_backend = TraceLogWriter(log_dir_path)

    sinks: list[Sink] = [
        JsonlSink(json_backend),
        ConsoleSink(console_backend),
        TraceLogSink(trace_backend),
    ]

    if vertex_experiment:
        if not vertex_project:
            raise ValueError(
                "vertex_project is required when vertex_experiment is set."
            )
        # Default to a job-id-embedding name (job-{job_id}-{timestamp}) so a
        # Vertex run is traceable to its tuning job; an explicit
        # vertex_run_name still wins.
        run_name = vertex_run_name or default_vertex_run_name(log_dir_path)
        ve_backend = VertexExperimentLogger(
            experiment=vertex_experiment,
            run_name=run_name,
            project=vertex_project,
            location=vertex_location,
            tensorboard=vertex_tensorboard,
        )
        sinks.insert(0, MLPlatformSink(ve_backend))

    if wandb_project:
        wb_run_name = wandb_run_name or log_dir_path.name
        wb_backend = WandbLogger(
            project=wandb_project,
            run_name=wb_run_name,
            log_dir=str(log_dir_path),
            entity=wandb_entity,
            mode=wandb_mode,
        )
        sinks.insert(0, MLPlatformSink(wb_backend))

    if extra_ml_platform_backends:
        for backend in extra_ml_platform_backends:
            sinks.append(
                backend if isinstance(backend, Sink) else MLPlatformSink(backend)
            )

    ml_logger = SinkDispatcher(sinks)

    if config is not None:
        ml_logger.log_hparams(config)

    if do_configure_logging_module:
        configure_logging_module(str(log_dir_path / "logs.log"))

    logger.info(f"Logging to: {log_dir_path}")
    return ml_logger


def configure_logging_module(path: str, level: int = logging.INFO) -> logging.Logger:
    """Configure logging to console (color) and file (plain), forcing override of prior config."""

    # ANSI escape codes for colors
    COLORS = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[95m",  # Magenta
    }
    RESET = "\033[0m"

    class ColorFormatter(logging.Formatter):
        """Colorized log formatter for console output that doesn't mutate record.levelname."""

        def format(self, record: logging.LogRecord) -> str:
            color = COLORS.get(record.levelname, "")
            # add a separate attribute for the colored level name
            record.levelname_colored = f"{color}{record.levelname}{RESET}"
            return super().format(record)

    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        ColorFormatter(
            "%(asctime)s %(name)s:%(lineno)d [%(levelname_colored)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S %z",
        )
    )

    class PlainFormatter(logging.Formatter):
        """Formatter that strips ANSI escape sequences for clean file output."""

        def __init__(self, fmt: str, datefmt: str | None = None):
            super().__init__(fmt, datefmt=datefmt)
            import re

            self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

        def format(self, record: logging.LogRecord) -> str:
            original_msg = super().format(record)
            return self.ansi_escape.sub("", original_msg)

    # File handler without colors
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(
        PlainFormatter(
            "%(asctime)s %(name)s:%(lineno)d [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S %z",
        )
    )

    # Force override like basicConfig(..., force=True)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Third destination: a per-rollout file chosen by the current context.
    # No-ops for records emitted outside a rollout, so this is purely additive
    # to the two handlers above.
    install_rollout_log_handler()

    # httpx/httpcore emit one INFO line per request; cap at WARNING to
    # avoid drowning the log in healthy-200 noise from tool-executor loops.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    debug_loggers = [
        n.strip() for n in os.environ.get("DEBUG_LOGGERS", "").split(",") if n.strip()
    ]
    for name in debug_loggers:
        logging.getLogger(name).setLevel(logging.DEBUG)
    if debug_loggers:
        # Confirmation line so it's clear from the log whether the knob is
        # actually live in this process.
        logger.info("DEBUG_LOGGERS active: DEBUG enabled for %s", debug_loggers)

    return root


def restore_iterator_state(
    resume_metrics_path: str, resume_from_train_step: int
) -> int | None:
    """Restores the iterator current index from a metrics file.

    Args:
        resume_metrics_path: Path to the metrics.jsonl file.
        resume_from_train_step: The train_step we are resuming from.

    Returns:
        The restored current index, or None if failed or not found.
    """
    if not resume_metrics_path:
        return None

    try:
        with open(resume_metrics_path, "r") as f:
            for line in f:
                data = json.loads(line)
                if int(data.get("step", -1)) == resume_from_train_step:
                    if "train/pipeline/iterator_current" in data:
                        current = int(data["train/pipeline/iterator_current"])
                        logger.info(
                            f"Restored train_iter.current to {current} from {resume_metrics_path} for train_step {resume_from_train_step}"
                        )
                        return current
    except Exception as e:
        logger.warning(
            f"Failed to restore iterator state from {resume_metrics_path}: {e}"
        )

    return None
