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

"""Registration + reclaim of abandoned ATIF runs.

The dangerous operation here is deleting another process's data, so most of
these tests are about the cases where reclaim must REFUSE.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from utils.run_registry import (
    RunRegistration,
    RunRegistry,
    is_alive,
    reclaim_abandoned,
)


def _reg(**kw) -> RunRegistration:
    base = dict(
        run_id="r1",
        job_dir="/nonexistent",
        gcs_prefix="gs://b/p",
        pid=os.getpid(),
        boot_id="boot-a",
        started_at=0.0,
    )
    base.update(kw)
    return RunRegistration(**base)


class TestLiveness:
    def test_our_own_pid_is_alive(self):
        assert is_alive(_reg(boot_id=""))

    def test_a_pid_from_another_boot_is_never_trusted(self):
        """After a reboot the same PID is some unrelated process.

        Without the boot check, a recycled PID makes a dead run look alive
        forever -- its tree then never gets reclaimed.
        """
        import utils.run_registry as rr

        real = rr._boot_id()
        if not real:
            pytest.skip("no boot_id on this platform")
        assert not is_alive(_reg(boot_id=real + "-different"))

    def test_an_unknown_pid_is_dead(self):
        assert not is_alive(_reg(pid=2**22, boot_id=""))

    def test_a_nonsense_pid_is_dead(self):
        assert not is_alive(_reg(pid=0, boot_id=""))


class TestRegistry:
    def test_register_then_read_back(self, tmp_path):
        reg = RunRegistry(tmp_path)
        out = reg.register(run_id="run-1", job_dir=tmp_path / "jd", gcs_prefix="gs://b/p")
        assert out is not None
        got = list(reg.all_registrations())
        assert [r.run_id for r in got] == ["run-1"]
        # The prefix is RECORDED, not re-derived later from the directory
        # name -- that is the whole reason registration exists.
        assert got[0].gcs_prefix == "gs://b/p"
        assert got[0].pid == os.getpid()

    def test_deregister_removes_it(self, tmp_path):
        reg = RunRegistry(tmp_path)
        reg.register(run_id="run-1", job_dir=tmp_path, gcs_prefix="gs://b/p")
        reg.deregister("run-1")
        assert list(reg.all_registrations()) == []

    def test_a_corrupt_registration_is_skipped_not_fatal(self, tmp_path):
        (tmp_path / "bad.json").write_text("{not json")
        reg = RunRegistry(tmp_path)
        reg.register(run_id="good", job_dir=tmp_path, gcs_prefix="gs://b/p")
        assert [r.run_id for r in reg.all_registrations()] == ["good"]

    def test_a_live_run_is_never_abandoned(self, tmp_path):
        jd = tmp_path / "jd"
        jd.mkdir()
        (jd / "f").write_text("x")
        reg = RunRegistry(tmp_path)
        reg.register(run_id="mine", job_dir=jd, gcs_prefix="gs://b/p")
        # Our own pid: alive, so not reclaimable however quiet the tree is.
        assert reg.abandoned(stale_after_s=0.0) == []

    def test_a_dead_run_with_a_BUSY_tree_is_left_alone(self, tmp_path):
        """Dead pid + recent writes means something else is still working.

        Reclaiming here would race whatever inherited the tree.
        """
        jd = tmp_path / "jd"
        jd.mkdir()
        (jd / "f").write_text("x")  # mtime = now
        reg = RunRegistry(tmp_path)
        path = tmp_path / "dead.json"
        path.write_text(
            _reg(run_id="dead", job_dir=str(jd), pid=2**22, boot_id="").to_json()
        )
        assert reg.abandoned(stale_after_s=3600.0) == []

    def test_a_dead_run_with_a_QUIET_tree_is_abandoned(self, tmp_path):
        jd = tmp_path / "jd"
        jd.mkdir()
        (jd / "f").write_text("x")
        reg = RunRegistry(tmp_path)
        (tmp_path / "dead.json").write_text(
            _reg(run_id="dead", job_dir=str(jd), pid=2**22, boot_id="").to_json()
        )
        got = reg.abandoned(stale_after_s=0.0)
        assert [r.run_id for r in got] == ["dead"]


class _FakeUploader:
    def __init__(self, result):
        self.result = result
        self.seen: list[Path] = []

    async def reclaim_tree(self, job_dir: Path):
        self.seen.append(job_dir)
        return self.result


class TestReclaim:
    async def test_a_clean_reclaim_drops_the_registration(self, tmp_path):
        jd = tmp_path / "jd"
        jd.mkdir()
        (jd / "f").write_text("x")
        reg = RunRegistry(tmp_path)
        (tmp_path / "dead.json").write_text(
            _reg(run_id="dead", job_dir=str(jd), pid=2**22, boot_id="").to_json()
        )
        up = _FakeUploader({"uploaded": 3, "deleted": 3, "failed": 0})
        stats = await reclaim_abandoned(
            reg, uploader_factory=lambda r: up, stale_after_s=0.0
        )
        assert stats["runs"] == 1 and stats["deleted"] == 3
        assert list(reg.all_registrations()) == []

    async def test_a_failed_upload_KEEPS_the_registration(self, tmp_path):
        """Something did not reach GCS, so the evidence must stay reclaimable.

        Dropping the registration here would strand those files forever: no
        later run would know to look at them.
        """
        jd = tmp_path / "jd"
        jd.mkdir()
        (jd / "f").write_text("x")
        reg = RunRegistry(tmp_path)
        (tmp_path / "dead.json").write_text(
            _reg(run_id="dead", job_dir=str(jd), pid=2**22, boot_id="").to_json()
        )
        up = _FakeUploader({"uploaded": 1, "deleted": 1, "failed": 2})
        await reclaim_abandoned(reg, uploader_factory=lambda r: up, stale_after_s=0.0)
        assert [r.run_id for r in reg.all_registrations()] == ["dead"]

    async def test_an_exploding_uploader_does_not_fail_the_caller(self, tmp_path):
        """Reclaim is housekeeping for the run that is STARTING.

        If it raised, a GCS outage would stop training from starting at all.
        """
        jd = tmp_path / "jd"
        jd.mkdir()
        (jd / "f").write_text("x")
        reg = RunRegistry(tmp_path)
        (tmp_path / "dead.json").write_text(
            _reg(run_id="dead", job_dir=str(jd), pid=2**22, boot_id="").to_json()
        )

        def boom(_r):
            raise RuntimeError("gcs is down")

        stats = await reclaim_abandoned(
            reg, uploader_factory=boom, stale_after_s=0.0
        )
        assert stats["runs"] == 0
        assert [r.run_id for r in reg.all_registrations()] == ["dead"]

    async def test_a_registration_whose_tree_is_gone_is_just_litter(self, tmp_path):
        reg = RunRegistry(tmp_path)
        (tmp_path / "dead.json").write_text(
            _reg(run_id="dead", job_dir=str(tmp_path / "gone"), pid=2**22, boot_id="").to_json()
        )
        up = _FakeUploader({"uploaded": 0, "deleted": 0, "failed": 0})
        await reclaim_abandoned(reg, uploader_factory=lambda r: up, stale_after_s=0.0)
        assert list(reg.all_registrations()) == []
        assert up.seen == []


class TestCli:
    """`python -m utils.run_registry` -- the ONLY thing that reclaims.

    Training runs deliberately do not: an earlier draft swept abandoned trees
    from the train/eval entry points, which made run B responsible for run
    A's data. Reclaim is now an operator action.
    """

    @staticmethod
    def _dead_run(tmp_path):
        reg_dir = tmp_path / "reg"
        job_dir = tmp_path / "recipe" / "deadjob"
        (job_dir / "trial-1").mkdir(parents=True)
        for i in range(3):
            (job_dir / "trial-1" / f"f{i}.json").write_text("{}")
        reg_dir.mkdir()
        (reg_dir / "dead.json").write_text(
            _reg(run_id="deadjob", job_dir=str(job_dir), pid=2**22, boot_id="").to_json()
        )
        return reg_dir, job_dir

    def test_listing_is_the_default_and_deletes_nothing(self, tmp_path, capsys):
        """The alternative deletes data, so you must ask for it explicitly."""
        from utils.run_registry import _cli

        reg_dir, job_dir = self._dead_run(tmp_path)
        rc = _cli(["--registry-dir", str(reg_dir), "--stale-after", "0"])
        assert rc == 0
        assert len(list(job_dir.rglob("*.json"))) == 3
        out = capsys.readouterr().out
        assert "abandoned runs : 1" in out
        assert "--reclaim" in out  # tells you how to actually do it

    def test_reclaim_uploads_and_deletes(self, tmp_path, monkeypatch):
        from utils import run_registry as rr

        reg_dir, job_dir = self._dead_run(tmp_path)
        seen: list = []

        class _Up:
            def __init__(self, **kw):
                self.prefix = kw.get("gcs_prefix")

            async def reclaim_tree(self, jd):
                seen.append((self.prefix, jd))
                for p in list(jd.rglob("*.json")):
                    p.unlink()
                return {"uploaded": 3, "deleted": 3, "failed": 0}

        monkeypatch.setattr(
            "utils.atif_disk_writer._GcsUploader", _Up, raising=True
        )
        rc = rr._cli(["--registry-dir", str(reg_dir), "--stale-after", "0", "--reclaim"])
        assert rc == 0
        assert list(job_dir.rglob("*.json")) == []
        # Each tree goes to the prefix ITS OWN registration recorded.
        assert seen[0][0] == "gs://b/p"

    def test_a_live_run_is_never_touched_by_the_cli(self, tmp_path, capsys):
        from utils.run_registry import RunRegistry, _cli

        job_dir = tmp_path / "recipe" / "livejob"
        job_dir.mkdir(parents=True)
        (job_dir / "f.json").write_text("{}")
        reg_dir = tmp_path / "reg"
        RunRegistry(reg_dir).register(
            run_id="livejob", job_dir=job_dir, gcs_prefix="gs://b/p"
        )
        rc = _cli(["--registry-dir", str(reg_dir), "--stale-after", "0", "--reclaim"])
        assert rc == 0
        assert (job_dir / "f.json").exists()
        assert "live runs      : 1" in capsys.readouterr().out

    def test_exit_code_is_nonzero_when_something_could_not_upload(
        self, tmp_path, monkeypatch
    ):
        """So a cron line reports a problem instead of silent success."""
        from utils import run_registry as rr

        reg_dir, _job_dir = self._dead_run(tmp_path)

        class _Up:
            def __init__(self, **kw):
                pass

            async def reclaim_tree(self, jd):
                return {"uploaded": 1, "deleted": 1, "failed": 2}

        monkeypatch.setattr("utils.atif_disk_writer._GcsUploader", _Up, raising=True)
        rc = rr._cli(["--registry-dir", str(reg_dir), "--stale-after", "0", "--reclaim"])
        assert rc == 1


class TestSandboxGuard:
    """Deleting the sandbox is the only op that leaves the ATIF tree.

    It removes a whole rsynced deploy (venv + code), driven by a `cwd` the
    dead process recorded, so every check here is a refusal by default.
    """

    @staticmethod
    def _valid(tmp_path):
        sb = tmp_path / "sandbox"
        jd = sb / "logs" / "atif" / "recipe" / "job"
        jd.mkdir(parents=True)
        return _reg(job_dir=str(jd), sandbox_dir=str(sb)), sb

    def test_a_well_formed_sandbox_is_allowed(self, tmp_path):
        from utils.run_registry import sandbox_is_safe_to_remove

        reg, _sb = self._valid(tmp_path)
        ok, why = sandbox_is_safe_to_remove(reg)
        assert ok, why

    @pytest.mark.parametrize(
        "sandbox,expect",
        [
            ("", "no sandbox_dir"),
            ("/", "shallow"),
            ("/tmp", "shallow"),
            ("relative/path", "not absolute"),
        ],
    )
    def test_dangerous_paths_are_refused(self, tmp_path, sandbox, expect):
        from utils.run_registry import sandbox_is_safe_to_remove

        reg, _ = self._valid(tmp_path)
        ok, why = sandbox_is_safe_to_remove(
            _reg(job_dir=reg.job_dir, sandbox_dir=sandbox)
        )
        assert not ok and expect in why

    def test_a_job_dir_outside_the_sandbox_is_refused(self, tmp_path):
        """The two paths then describe different things; trust neither."""
        from utils.run_registry import sandbox_is_safe_to_remove

        _reg_ok, sb = self._valid(tmp_path)
        ok, why = sandbox_is_safe_to_remove(
            _reg(job_dir="/var/tmp/somewhere-else", sandbox_dir=str(sb))
        )
        assert not ok and "not inside" in why

    def test_the_reclaimers_own_tree_is_refused(self, tmp_path, monkeypatch):
        """Never delete the checkout the CLI is running from."""
        from utils.run_registry import sandbox_is_safe_to_remove

        reg, sb = self._valid(tmp_path)
        inner = sb / "sub"
        inner.mkdir()
        monkeypatch.chdir(inner)
        ok, why = sandbox_is_safe_to_remove(reg)
        assert not ok and "own working tree" in why


class TestRunLogDiscovery:
    def test_a_redirected_stdout_is_found(self, tmp_path, monkeypatch):
        """remote-run.sh never tells us the path; we read it off fd 1."""
        import os

        from utils.run_registry import current_run_log

        log = tmp_path / "run.out"
        log.write_text("hello")
        fd = os.open(log, os.O_WRONLY)
        saved = os.dup(1)
        try:
            os.dup2(fd, 1)
            assert current_run_log() == str(log)
        finally:
            os.dup2(saved, 1)
            os.close(saved)
            os.close(fd)

    def test_a_pipe_or_tty_yields_nothing_to_upload(self):
        from utils.run_registry import current_run_log

        # Under pytest, fd 1 is a capture pipe -- not a regular file.
        assert current_run_log() == "" or Path(current_run_log()).is_file()
