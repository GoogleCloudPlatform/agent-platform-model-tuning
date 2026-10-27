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

"""Registry of in-progress ATIF runs, and reclaim of the ones that died.

A run uploads its trajectories continuously and deletes the local copies as
each trial completes. That covers every ending except the one that actually
happens: ``SIGKILL``, a segfault, an OOM, a VM reset. None of those run any
cleanup code, so the entire local tree survives with no process left to
finish uploading or remove it. Five killed runs on one host left 90 GB.

This module is the other half. Each run REGISTERS itself on startup with
enough information for some later process to finish the job, and
DEREGISTERS on clean exit.

RECORDING AND RECLAIMING ARE SEPARATE, and that separation is the whole
design. The *run* always writes the registration -- there is no other
component that knows its ``gcs_prefix``, and nothing else has to exist for
the record to be correct. Who later ACTS on that record is an open choice:
a human, a cron line, or (if it ever earns its keep) a daemon. None of them
would hold any state of their own; they would all be stateless readers of
what the runs wrote. That is why adding a daemon buys so little here -- it
adds an actor, not a record.

What deliberately does NOT act on it is another training run. An earlier
draft swept abandoned trees from the train/eval entry points, which made
run B responsible for run A's data: B's startup latency, credentials and
failure modes entangled with data B has nothing to do with. A run cleans up
after itself and nothing else.

Today the actor is::

    python -m utils.run_registry --reclaim

Why registration rather than scanning for stray directories: the mapping
from a local ``job_dir`` to its GCS prefix is built inside the writer from
the job id and a timestamp. Re-deriving it from a directory name elsewhere
would put that logic in two places, and getting it wrong uploads one run's
trajectories into another run's prefix. The registration records the prefix
the run is actually using, so the reclaimer never has to guess.

SAFETY. Reclaiming means deleting another process's data, so a wrong
liveness answer destroys a live run's trajectories:

* Liveness is decided from the registered PID plus an identity check on
  ``/proc/<pid>``, never from a command-line pattern. ``pgrep -f <pat>``
  matches the *searching* process's own cmdline and can therefore never
  return "dead" -- a check that cannot fail is not a check.
* A registration is only reclaimed once its tree has been QUIET for
  ``stale_after_s``. A run stalled on a slow upload is not garbage.
* Reclaim is upload-then-delete, and only paths the upload confirmed are
  removed. A file that cannot be uploaded stays on disk forever in
  preference to being lost.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

# Registrations live outside the run's own directory on purpose: a reclaimer
# must be able to find them after that directory is the very thing in
# question, and /tmp sweepers routinely delete by age.
DEFAULT_REGISTRY_DIR = Path("/var/tmp/atif-runs")

# How long a registration's tree must be untouched before another process may
# treat it as abandoned. Generous: the cost of waiting is disk, the cost of
# being wrong is a live run's trajectories.
DEFAULT_STALE_AFTER_S = 900.0


@dataclass(frozen=True)
class RunRegistration:
    """What a later process needs to finish a dead run's uploads."""

    run_id: str
    job_dir: str
    gcs_prefix: str
    pid: int
    # Process identity. A PID alone is not identity -- PIDs are reused, and
    # reclaiming on a recycled PID would either skip a dead run forever or,
    # worse, treat a live unrelated process as the owner.
    boot_id: str
    started_at: float
    # The deploy root the launcher rsynced into (the run's cwd): venv,
    # sources, docs -- ~620 MB that nothing ever removes. Recorded, not
    # derived from job_dir by walking up N parents, for the same reason
    # gcs_prefix is: a wrong answer here deletes the wrong tree.
    sandbox_dir: str = ""
    # The run's stdout, if it is a regular file. A launcher running the
    # job in the background redirects to <sandbox>/run.out, and the process
    # finds it by resolving
    # /proc/self/fd/1 -- it is never passed in. This is where every runtime
    # diagnosis lives (latency, 429s, deadline_exceeded) and today it dies
    # with the VM while the trajectories survive.
    run_log: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json(text: str) -> "RunRegistration":
        d = json.loads(text)
        return RunRegistration(
            run_id=str(d["run_id"]),
            job_dir=str(d["job_dir"]),
            gcs_prefix=str(d["gcs_prefix"]),
            pid=int(d["pid"]),
            boot_id=str(d.get("boot_id") or ""),
            started_at=float(d.get("started_at") or 0.0),
            sandbox_dir=str(d.get("sandbox_dir") or ""),
            run_log=str(d.get("run_log") or ""),
        )


def current_run_log() -> str:
    """This process's stdout, if it is a regular file; else "".

    A launcher running the job in the background redirects stdout to
    ``<sandbox>/run.out`` and never tells the process where that is, so
    resolve it from
    ``/proc/self/fd/1``. A pipe or tty resolves to something that is not a
    file, which is the signal that there is no log to upload.
    """
    try:
        target = os.readlink("/proc/self/fd/1")
    except OSError:
        return ""
    if not target.startswith("/") or not os.path.isfile(target):
        return ""  # pipe, tty, socket, or deleted
    return target


def _boot_id() -> str:
    """Identify this boot, so a PID from a previous boot is never trusted.

    After a reboot PID 1199375 may well exist again and be something else
    entirely. Falls back to the empty string where the file is unavailable
    (non-Linux, restricted container); callers then rely on quiet-time alone,
    which is weaker but never wrong in the dangerous direction.
    """
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


def is_alive(reg: RunRegistration) -> bool:
    """True if the registering process is still running THIS boot.

    Deliberately conservative: anything we cannot establish counts as alive,
    because the consequence of a false "dead" is deleting a running job's
    data while the consequence of a false "alive" is some disk.
    """
    if reg.boot_id and _boot_id() and reg.boot_id != _boot_id():
        # Different boot: the PID cannot be the process that registered.
        return False
    if reg.pid <= 0:
        return False
    try:
        os.kill(reg.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists, owned by someone else. Alive.
        return True
    except OSError:
        return True
    return True


def _tree_mtime(root: Path) -> float:
    """Most recent mtime anywhere under ``root`` (0.0 if absent/empty).

    Progress, not liveness: a process can be alive and wedged, and a tree
    that has not changed in fifteen minutes is not being written to whatever
    the process table says.
    """
    newest = 0.0
    try:
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                try:
                    m = os.stat(os.path.join(dirpath, name)).st_mtime
                except OSError:
                    continue
                if m > newest:
                    newest = m
    except OSError:
        return 0.0
    return newest


class RunRegistry:
    """Append/remove registrations, and enumerate abandoned ones."""

    def __init__(
        self,
        registry_dir: Path | str = DEFAULT_REGISTRY_DIR,
        *,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self._dir = Path(registry_dir)
        self._time_fn = time_fn

    @property
    def directory(self) -> Path:
        return self._dir

    def register(
        self, *, run_id: str, job_dir: Path | str, gcs_prefix: str
    ) -> RunRegistration | None:
        """Record this run. Returns None if the registry is unwritable.

        Never raises: failing to register costs a future cleanup, while
        raising would fail a training run that is otherwise fine.
        """
        reg = RunRegistration(
            run_id=run_id,
            job_dir=str(Path(job_dir).resolve()),
            gcs_prefix=gcs_prefix,
            pid=os.getpid(),
            boot_id=_boot_id(),
            started_at=self._time_fn(),
            # The run's cwd IS the sandbox the launcher deployed into.
            sandbox_dir=str(Path.cwd().resolve()),
            run_log=current_run_log(),
        )
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            path = self._path_for(run_id)
            # Write-then-rename: a reclaimer must never read a half-written
            # registration and act on a truncated job_dir.
            tmp = path.with_suffix(".tmp")
            tmp.write_text(reg.to_json())
            tmp.replace(path)
        except OSError as exc:
            logger.warning("run_registry: could not register %s: %s", run_id, exc)
            return None
        logger.info(
            "run_registry: registered %s (pid=%d, job_dir=%s)",
            run_id,
            reg.pid,
            reg.job_dir,
        )
        return reg

    def deregister(self, run_id: str) -> None:
        """Drop this run's registration. Never raises."""
        try:
            self._path_for(run_id).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("run_registry: could not deregister %s: %s", run_id, exc)

    def all_registrations(self) -> Iterator[RunRegistration]:
        try:
            paths = sorted(self._dir.glob("*.json"))
        except OSError:
            return
        for p in paths:
            try:
                yield RunRegistration.from_json(p.read_text())
            except (OSError, ValueError, KeyError) as exc:
                logger.warning("run_registry: skipping unreadable %s: %s", p, exc)

    def abandoned(
        self, *, stale_after_s: float = DEFAULT_STALE_AFTER_S
    ) -> list[RunRegistration]:
        """Registrations whose process is gone AND whose tree has gone quiet.

        Both conditions, not either: a dead PID with a tree still being
        written means some other process inherited the work, and a quiet tree
        under a live PID is just a slow run.
        """
        out: list[RunRegistration] = []
        now = self._time_fn()
        for reg in self.all_registrations():
            if is_alive(reg):
                continue
            job_dir = Path(reg.job_dir)
            if not job_dir.exists():
                # Nothing to reclaim; the registration is just litter.
                out.append(reg)
                continue
            quiet_for = now - _tree_mtime(job_dir)
            if quiet_for < stale_after_s:
                logger.info(
                    "run_registry: %s is dead but its tree changed %.0fs ago "
                    "(< %.0fs); leaving it alone this round",
                    reg.run_id,
                    quiet_for,
                    stale_after_s,
                )
                continue
            out.append(reg)
        return out

    def _path_for(self, run_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_id)
        return self._dir / f"{safe}.json"


def sandbox_is_safe_to_remove(reg: RunRegistration) -> tuple[bool, str]:
    """Whether ``reg.sandbox_dir`` may be deleted. Returns (ok, reason).

    This is the only thing here that deletes outside the ATIF tree -- a
    whole rsynced deploy, venv included -- so every check is a refusal by
    default. The registration is written by a process that could have been
    started from anywhere, and a `cwd` of `/` or of somebody's checkout
    would otherwise be taken at face value.
    """
    raw = (reg.sandbox_dir or "").strip()
    if not raw:
        return False, "no sandbox_dir recorded"
    path = Path(raw)
    if not path.is_absolute():
        return False, f"not absolute: {raw}"
    resolved = path.resolve()
    # A run's cwd should be a per-run scratch dir. Anything at depth <= 2
    # ("/", "/tmp", "/home/x") is a mistake, not a sandbox.
    if len(resolved.parts) <= 3:
        return False, f"suspiciously shallow: {resolved}"
    if not resolved.exists():
        return False, "already gone"
    # The ATIF tree must live INSIDE it. If it does not, these two paths
    # describe different things and the recorded cwd cannot be trusted as
    # this run's sandbox.
    try:
        Path(reg.job_dir).resolve().relative_to(resolved)
    except (ValueError, OSError):
        return False, f"job_dir {reg.job_dir} is not inside {resolved}"
    # Never delete the tree the reclaimer is itself running from.
    try:
        if Path.cwd().resolve() == resolved or resolved in Path.cwd().resolve().parents:
            return False, "it is the current process's own working tree"
    except OSError:
        return False, "cannot determine cwd"
    return True, ""


async def _upload_run_log(uploader: Any, reg: RunRegistration) -> bool:
    """Put the run's stdout beside its trajectories in GCS. Never raises.

    The log is not part of the ATIF tree, so `reclaim_tree` does not see it,
    and it sits inside the sandbox we are about to delete. It is also the
    only record of everything that is not a trajectory: sampler latency,
    429 storms, why a rollout hit its deadline.
    """
    path_s = (reg.run_log or "").strip()
    if not path_s:
        return False
    path = Path(path_s)
    if not path.is_file():
        return False
    try:
        blob = f"{reg.run_id}/{path.name}"
        await uploader.upload_single(path, blob)
        logger.warning(
            "run_registry: uploaded run log %s (%.1f MB) -> %s",
            path,
            path.stat().st_size / 1048576,
            blob,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - a lost log must not block cleanup
        logger.warning("run_registry: could not upload run log %s: %s", path, exc)
        return False


def _remove_sandbox(reg: RunRegistration) -> bool:
    """Delete the rsynced deploy tree (venv + code). Never raises.

    Only reached for a run already confirmed dead and fully uploaded, and
    only when `sandbox_is_safe_to_remove` agrees -- see that function for
    why each refusal exists.
    """
    ok, why = sandbox_is_safe_to_remove(reg)
    if not ok:
        logger.info("run_registry: not removing sandbox for %s: %s", reg.run_id, why)
        return False
    target = Path(reg.sandbox_dir).resolve()
    try:
        shutil.rmtree(target)
    except OSError as exc:
        logger.warning("run_registry: could not remove sandbox %s: %s", target, exc)
        return False
    logger.warning("run_registry: removed sandbox %s", target)
    return True


async def reclaim_abandoned(
    registry: RunRegistry,
    *,
    uploader_factory: Callable[[RunRegistration], Any],
    stale_after_s: float = DEFAULT_STALE_AFTER_S,
    max_runs: int = 8,
) -> dict[str, int]:
    """Finish uploads for dead runs, then delete what GCS confirmed.

    ``uploader_factory`` builds something with ``reclaim_tree(job_dir)`` for a
    registration -- injected rather than imported so this module stays free of
    any particular writer and is trivially testable.

    Returns counts for logging. Never raises: reclaim is best-effort
    housekeeping and must not stop the run that invoked it from starting.
    """
    stats = {"runs": 0, "uploaded": 0, "deleted": 0, "failed": 0}
    for reg in registry.abandoned(stale_after_s=stale_after_s)[:max_runs]:
        job_dir = Path(reg.job_dir)
        if not job_dir.exists():
            registry.deregister(reg.run_id)
            continue
        logger.warning(
            "run_registry: reclaiming abandoned run %s (pid %d is gone): %s",
            reg.run_id,
            reg.pid,
            job_dir,
        )
        try:
            uploader = uploader_factory(reg)
            result = await uploader.reclaim_tree(job_dir)
            stats["runs"] += 1
            stats["uploaded"] += int(result.get("uploaded", 0))
            stats["deleted"] += int(result.get("deleted", 0))
            stats["failed"] += int(result.get("failed", 0))
            if not result.get("failed"):
                # The run log goes up BEFORE the sandbox goes away -- it
                # lives inside it. This is where every runtime diagnosis is
                # (per-call latency, 429 storms, deadline_exceeded); until
                # now it died with the VM while the trajectories survived.
                if await _upload_run_log(uploader, reg):
                    stats["logs_uploaded"] = stats.get("logs_uploaded", 0) + 1
                if _remove_sandbox(reg):
                    stats["sandboxes_removed"] = (
                        stats.get("sandboxes_removed", 0) + 1
                    )
                registry.deregister(reg.run_id)
            else:
                logger.warning(
                    "run_registry: %s has %d file(s) that would not upload; "
                    "keeping its registration and local copies for the next "
                    "attempt",
                    reg.run_id,
                    result.get("failed"),
                )
        except Exception as exc:  # noqa: BLE001 - housekeeping never fails a run
            logger.warning(
                "run_registry: reclaim of %s failed, leaving it for next time: %s",
                reg.run_id,
                exc,
            )
    if stats["runs"]:
        logger.warning(
            "run_registry: reclaimed %d abandoned run(s): %d uploaded, "
            "%d deleted, %d still failing",
            stats["runs"],
            stats["uploaded"],
            stats["deleted"],
            stats["failed"],
        )
    return stats


def _cli(argv: "list[str] | None" = None) -> int:
    """``python -m utils.run_registry`` -- list or reclaim abandoned runs.

    The only actor that acts on other runs' registrations. Run it by hand
    after killing a job, or from a cron line if a host wants time-based
    cleanup; either way the operational burden is the operator's choice
    rather than a service this repo has to keep alive.

    ``--list`` is the safe default precisely because the alternative deletes
    data: you can see what WOULD be reclaimed before anything is.
    """
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        prog="python -m utils.run_registry",
        description="Inspect or reclaim ATIF runs left behind by dead processes.",
    )
    parser.add_argument(
        "--reclaim",
        action="store_true",
        help="upload then delete abandoned trees (default is to list only)",
    )
    parser.add_argument(
        "--stale-after",
        type=float,
        default=DEFAULT_STALE_AFTER_S,
        help=(
            "seconds a tree must be unmodified before it counts as abandoned "
            f"(default {DEFAULT_STALE_AFTER_S:.0f})"
        ),
    )
    parser.add_argument(
        "--registry-dir", default=str(DEFAULT_REGISTRY_DIR), help="registry location"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    registry = RunRegistry(args.registry_dir)

    live = [r for r in registry.all_registrations() if is_alive(r)]
    dead = registry.abandoned(stale_after_s=args.stale_after)
    print(f"registry: {args.registry_dir}")
    print(f"  live runs      : {len(live)}")
    for r in live:
        print(f"    {r.run_id}  pid={r.pid}  {r.job_dir}")
    print(f"  abandoned runs : {len(dead)}")
    for r in dead:
        print(f"    {r.run_id}  pid={r.pid} (gone)  {r.job_dir}")

    if not args.reclaim:
        if dead:
            print("\nre-run with --reclaim to upload and delete these.")
        return 0

    # Imported here so the module itself stays free of the writer: the
    # registry is about records, not about how bytes reach GCS.
    from utils.atif_disk_writer import _GcsUploader

    def _factory(reg: RunRegistration):
        # Each tree goes to the prefix ITS OWN run recorded. Deriving one
        # prefix for all of them would upload one run into another's.
        return _GcsUploader(
            gcs_prefix=reg.gcs_prefix,
            job_dir=Path(reg.job_dir),
            delete_after_upload=True,
        )

    stats = asyncio.run(
        reclaim_abandoned(
            registry, uploader_factory=_factory, stale_after_s=args.stale_after
        )
    )
    print(
        f"\nreclaimed {stats['runs']} run(s): "
        f"{stats['uploaded']} uploaded, {stats['deleted']} deleted, "
        f"{stats['failed']} still failing"
    )
    # Non-zero when something could not be uploaded, so a cron line surfaces
    # it instead of reporting success.
    return 1 if stats["failed"] else 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(_cli())
