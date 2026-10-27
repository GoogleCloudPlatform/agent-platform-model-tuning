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

"""Tests for core.agent + core.metric_aggregators (post-ScoredTrajectory migration)."""

import asyncio
import dataclasses
import math

import pytest

from core import (
    agent,
    atif_extras,
    atif_trajectory,
    metric_aggregators,
    protocols,
    single_step_agent,
)
from core.atif_trajectory import (
    ScoredTrajectory,
    Reward,
    StepStats,
    StepTimings,
    TrajectoryStep as AtifTrajectoryStep,
    Step,
)
from core.metric_spec import MetricValue
from core.rollout_status import RolloutStatus
from core.task_data import TaskData
from core.atif_trajectory import Agent as AtifAgent, Trajectory as AtifTrajectory
from tests.unit._sample_result_factories import make_sample_result


@dataclasses.dataclass
class _TestTaskData(TaskData):
    """Lightweight TaskData subclass for tests that don't care about payload."""

    pass


def _test_agent_kwargs() -> dict:
    """Common kwargs for constructing a test SingleStepAgent."""
    return dict(
        recipe_name="test_recipe",
        tuning_job_id="tj-test",
        extract_prompt_text=lambda td: "test prompt",
    )


def _agent_trajectory(
    *,
    reward: float,
    is_correct: bool | None = None,
    status: RolloutStatus = RolloutStatus.OK,
    stats: StepStats | None = None,
    n_steps: int = 1,
) -> ScoredTrajectory:
    """Build a ScoredTrajectory with n agent steps (all sharing `stats`)."""
    us = Step(step_id=1, source="user", message="p")
    agent_steps: list = []
    for i in range(n_steps):
        s = AtifTrajectoryStep.from_wire(
            step_id=2 + i,
            generation="g",
            stats=stats if stats is not None else StepStats(),
        )
        agent_steps.append(s)
    traj = AtifTrajectory(
        schema_version="ATIF-v1.7",
        session_id="test-session",
        agent=AtifAgent(name="t", version="0", model_name="m"),
        steps=[us, *agent_steps],
    )
    atif_extras.set_rollout_status(traj, status)
    if is_correct is not None:
        atif_extras.set_is_correct(traj, is_correct)
    return ScoredTrajectory(trajectory=traj, reward=Reward(reward=reward))


def _traj_with_train_step_ids(ids: list[int | None]) -> ScoredTrajectory:
    """ScoredTrajectory with one agent step per entry in ``ids``; each step's
    ``train_step_id`` is set to that entry (``None`` => step carries no
    train_step_id). Lets a single trajectory span multiple policy steps so the
    count-weighted mean can be exercised within one trajectory."""
    us = Step(step_id=1, source="user", message="p")
    agent_steps: list = []
    for i, ts_id in enumerate(ids):
        stats = StepStats(train_step_id=ts_id) if ts_id is not None else StepStats()
        agent_steps.append(
            AtifTrajectoryStep.from_wire(step_id=2 + i, generation="g", stats=stats)
        )
    traj = AtifTrajectory(
        schema_version="ATIF-v1.7",
        session_id="test-session",
        agent=AtifAgent(name="t", version="0", model_name="m"),
        steps=[us, *agent_steps],
    )
    atif_extras.set_rollout_status(traj, RolloutStatus.OK)
    return ScoredTrajectory(trajectory=traj, reward=Reward(reward=0.0))


class TestSamplerPolicyStepMetricAggregator:
    def test_empty_returns_empty(self):
        assert metric_aggregators.SamplerPolicyStepMetricAggregator()([]) == {}

    def test_no_train_step_id_returns_empty(self):
        assert (
            metric_aggregators.SamplerPolicyStepMetricAggregator()(
                [_agent_trajectory(reward=0.0)]
            )
            == {}
        )

    def test_policy_step_weighted_across_trajectories(self):
        trajs = [
            _agent_trajectory(reward=0.0, stats=StepStats(train_step_id=42)),
            _agent_trajectory(reward=0.0, stats=StepStats(train_step_id=42)),
            _agent_trajectory(reward=0.0, stats=StepStats(train_step_id=43)),
            _agent_trajectory(reward=0.0),  # no train_step_id
        ]
        m = metric_aggregators.SamplerPolicyStepMetricAggregator()(trajs)
        # 3 agent steps carry a train_step_id (4th trajectory has none);
        # weight = 3, value = (42+42+43)/3.
        assert m == {
            "sampler/policy_step": MetricValue.of_weighted((42 + 42 + 43) / 3, 3)
        }

    def test_sampler_policy_step_aggregator_emits_correct_key(self):
        """Emits sampler/policy_step (not avg_train_step_id) and computes the
        count-weighted mean of train_step_id across all agent-source steps.

        2 trajectories: first has 3 steps at train_step_id [10, 12, 14];
        second has 2 steps at [16, 18]. Mean = (10+12+14+16+18) / 5 = 14.0,
        weight = 5 (total agent steps with train_step_id set)."""
        trajectories = [
            _traj_with_train_step_ids([10, 12, 14]),
            _traj_with_train_step_ids([16, 18]),
        ]
        agg = metric_aggregators.SamplerPolicyStepMetricAggregator()
        assert agg(trajectories) == {
            "sampler/policy_step": MetricValue.of_weighted(14.0, 5)
        }

    def test_sampler_policy_step_aggregator_empty_when_no_train_step_ids(self):
        """When no step has a train_step_id (e.g. a gemini_api backend that
        doesn't emit one), emit empty dict -- no metric is better than a fake
        zero."""
        trajectories = [_traj_with_train_step_ids([None, None])]
        agg = metric_aggregators.SamplerPolicyStepMetricAggregator()
        assert agg(trajectories) == {}


class TestFinishReasonMetricAggregator:
    def test_empty_returns_empty(self):
        assert metric_aggregators.FinishReasonMetricAggregator()([]) == {}

    def test_no_finish_reason_returns_empty(self):
        assert (
            metric_aggregators.FinishReasonMetricAggregator()(
                [_agent_trajectory(reward=0.0)]
            )
            == {}
        )

    def test_counts_by_finish_reason(self):
        trajs = [
            _agent_trajectory(reward=0.0, stats=StepStats(finish_reason="STOP")),
            _agent_trajectory(reward=0.0, stats=StepStats(finish_reason="STOP")),
            _agent_trajectory(reward=0.0, stats=StepStats(finish_reason="MAX_TOKENS")),
            _agent_trajectory(reward=0.0),
        ]
        m = metric_aggregators.FinishReasonMetricAggregator()(trajs)
        # Keys use the sampler/finish_reason/* prefix (snake_case, sampler
        # bucket); values are of_ratio(count, total_agent_steps).
        # 4 trajectories -> total = 4.
        assert m["sampler/finish_reason/STOP"] == MetricValue.of_ratio(2, 4)
        assert m["sampler/finish_reason/MAX_TOKENS"] == MetricValue.of_ratio(1, 4)
        assert m["sampler/finish_reason/OTHER"] == MetricValue.of_ratio(1, 4)
        assert not any(k.startswith("finishReason/") for k in m)


class TestCoreRewardMetricAggregator:
    def test_typed_signal_wins_over_reward(self):
        t = _agent_trajectory(reward=-1.0, is_correct=True)
        m = metric_aggregators.CoreRewardMetricAggregator()([t])
        assert m["learning/sampling_accuracy"].value == pytest.approx(1.0)
        assert m["learning/pass_at_k"].value == pytest.approx(1.0)

    def test_falls_back_to_reward_positive(self):
        t = _agent_trajectory(reward=0.5)
        m = metric_aggregators.CoreRewardMetricAggregator()([t])
        assert m["learning/sampling_accuracy"].value == pytest.approx(1.0)

    def test_typed_is_correct_false_overrides_positive_reward(self):
        t = _agent_trajectory(reward=1.0, is_correct=False)
        m = metric_aggregators.CoreRewardMetricAggregator()([t])
        assert m["learning/sampling_accuracy"].value == pytest.approx(0.0)


class _FakeSampler(protocols.Sampler):
    def __init__(self, results):
        self._results = results

    async def sample_n(
        self,
        prompt,
        num_samples=1,
        **kwargs,
    ):
        return list(self._results[:num_samples])


from google.cloud.aiplatform_v1.types import (  # noqa: E402
    Content as _Content,
    GenerateContentRequest as _GenerateContentRequest,
)


def _prompt_gcr(text: str) -> _GenerateContentRequest:
    """Wrap a plain string prompt into a minimal GenerateContentRequest.

    Real recipe envs return a fully-formed GCR from
    ``get_prompt()``; test envs used to return a bare string. The refactor
    tightened ``Env.get_prompt() -> GenerateContentRequest``, so tests
    must wrap.
    """
    return _GenerateContentRequest(
        contents=[_Content(role="user", parts=[{"text": text}])]
    )


class _FakeEnv(protocols.Env):
    def __init__(self, prompt="p", reward=1.0):
        self._prompt = prompt
        self._reward = reward

    def get_prompt(self):
        return _prompt_gcr(self._prompt)

    async def step(self, action: protocols.Action, *, step_count: int = 0):
        del step_count
        return protocols.EnvStepResult(
            function_responses=[], reward=self._reward, done=True
        )


class TestMetricAggregatorsComposition:
    @staticmethod
    def _make_agent(metric_aggregators_arg):
        return single_step_agent.SingleStepAgent(
            sampler=_FakeSampler([]),
            env_factory=lambda _: _FakeEnv(),
            **_test_agent_kwargs(),
            metric_aggregators=metric_aggregators_arg,
        )

    def test_default_when_kwarg_omitted(self):
        a = single_step_agent.SingleStepAgent(
            sampler=_FakeSampler([]),
            env_factory=lambda _: _FakeEnv(),
            **_test_agent_kwargs(),
        )
        assert a.metric_aggregators is metric_aggregators.DEFAULT_METRIC_AGGREGATORS

    def test_extra_merges_after_defaults(self):
        def _extra(trajs):
            return {"my_recipe/foo": 42.0}

        a = self._make_agent(metric_aggregators.DEFAULT_METRIC_AGGREGATORS + [_extra])
        m = a._compute_group_metrics(
            [
                _agent_trajectory(reward=1.0),
                _agent_trajectory(reward=0.0),
            ]
        )
        from core.metric_spec import MetricValue as _MV

        assert m["my_recipe/foo"] == _MV.of_scalar(42.0)
        assert "learning/reward/mean" in m

    def test_empty_list_yields_empty(self):
        a = self._make_agent([])
        assert (
            a._compute_group_metrics(
                [
                    _agent_trajectory(reward=1.0),
                    _agent_trajectory(reward=0.0),
                ]
            )
            == {}
        )

    def test_same_stage_collision_raises(self):
        """Two aggregators emitting the same key in ONE group is now a same-
        stage collision rather than the old silent last-writer-wins.
        This is the guard that catches double-emit bugs."""

        def _override(trajs):
            return {"learning/reward/mean": 99.0}

        a = self._make_agent(
            [metric_aggregators.CoreRewardMetricAggregator(), _override]
        )
        import pytest as _pytest

        with _pytest.raises(ValueError, match="collision"):
            a._compute_group_metrics(
                [
                    _agent_trajectory(reward=1.0),
                    _agent_trajectory(reward=0.0),
                ]
            )

    def test_bare_float_wrapped_as_scalar_metricvalue_and_metricvalue_passes_through(
        self,
    ):
        from core.metric_spec import MetricValue

        def _bare(trajs):
            return {"learning/pass_at_k": 1.0}  # bare float (legacy shape)

        def _mv(trajs):
            return {"learning/sampling_accuracy": MetricValue.of_scalar(0.5)}

        a = self._make_agent([_bare, _mv])
        merged = a._compute_group_metrics(
            [_agent_trajectory(reward=1.0), _agent_trajectory(reward=0.0)]
        )
        assert merged["learning/pass_at_k"] == MetricValue.of_scalar(1.0)
        assert merged["learning/sampling_accuracy"] == MetricValue.of_scalar(0.5)


class TestAgentABC:
    def test_cannot_instantiate_without_all_abstract_methods(self):
        with pytest.raises(TypeError):
            agent.Agent(sampler=None, env_factory=None)  # type: ignore[abstract]


class TestSingleStepAgentInit:
    def test_missing_required_kwargs_raises(self):
        with pytest.raises(TypeError):
            single_step_agent.SingleStepAgent(
                sampler=_FakeSampler([]), env_factory=lambda _: _FakeEnv()
            )  # type: ignore[call-arg]


class TestBaseAgentAggregationSeam:
    """The aggregation seam (metric_aggregators +
    _compute_group_metrics) lives on the base ``Agent``, not on
    ``SingleStepAgent``. This lets other Agent subclasses (with bespoke
    multi-turn rollouts that can't subclass SingleStepAgent) share the
    same aggregation path."""

    @staticmethod
    def _bare_agent_cls():
        class _BareAgent(agent.Agent):
            async def _rollout_internal(self, task_data, ctx):  # pragma: no cover
                raise NotImplementedError

        return _BareAgent

    def test_base_agent_owns_aggregation_seam(self):
        """A bare Agent subclass (not SingleStepAgent) inherits the aggregation
        seam: metric_aggregators, _compute_group_metrics. Proves the seam moved
        up from SingleStepAgent."""
        bare_cls = self._bare_agent_cls()
        a = bare_cls(
            sampler=object(),
            env_factory=lambda td: object(),
            recipe_name="test",
            tuning_job_id="tj-test",
            metric_aggregators=[lambda _t: {"recipe/x/foo": 1.0}],
        )
        assert callable(a._compute_group_metrics)
        result = a._compute_group_metrics([])
        # bare-float 1.0 is wrapped as MetricValue.of_scalar
        assert result == {"recipe/x/foo": MetricValue.of_scalar(1.0)}

    def test_base_agent_metric_aggregators_defaults_to_framework_defaults(self):
        """When metric_aggregators is omitted, the base Agent uses
        DEFAULT_METRIC_AGGREGATORS (same default SingleStepAgent had)."""
        bare_cls = self._bare_agent_cls()
        a = bare_cls(
            sampler=object(),
            env_factory=lambda td: object(),
            recipe_name="test",
            tuning_job_id="tj-test",
        )
        assert a.metric_aggregators is metric_aggregators.DEFAULT_METRIC_AGGREGATORS

    def test_single_step_agent_still_has_seam_after_move(self):
        """Regression: SingleStepAgent still exposes the seam (via inheritance)
        after the move; the aggregators kwarg still works."""
        a = single_step_agent.SingleStepAgent(
            sampler=_FakeSampler([]),
            env_factory=lambda _: _FakeEnv(),
            **_test_agent_kwargs(),
            metric_aggregators=[lambda _t: {"recipe/word_puzzle/x": 1.0}],
        )
        result = a._compute_group_metrics([])
        assert result == {"recipe/word_puzzle/x": MetricValue.of_scalar(1.0)}


def _agent_steps_of(g: ScoredTrajectory):
    return [s for s in g.trajectory.steps if s.source == "agent"]


class TestSingleStepAgentGroupRollout:
    async def test_emits_base_metrics(self):
        results = [
            make_sample_result(text="a", candidate_id="c1"),
            make_sample_result(text="b", candidate_id="c2"),
        ]
        a = single_step_agent.SingleStepAgent(
            sampler=_FakeSampler(results),
            env_factory=lambda _: _FakeEnv(reward=1.0),
            **_test_agent_kwargs(),
        )
        group, m = await a.group_rollout(
            task_data=_TestTaskData(task_id="t1"), num_samples=2
        )
        assert len(group.scored_trajectories) == 2
        # _compute_group_metrics returns dict[str, MetricValue], and
        # reward/mean is a weighted pair (group_mean, n_trajectories).
        assert m["learning/reward/mean"].weighted_pair == (pytest.approx(1.0), 2)
        assert m["learning/pass_at_k"].value == 1.0
        assert m["learning/sampling_accuracy"].value == pytest.approx(1.0)

    async def test_typed_is_correct_takes_precedence_over_reward_sign(self):
        class _EnvWithIsCorrect(protocols.Env):
            def get_prompt(self):
                return _prompt_gcr("p")

            async def step(self, action: protocols.Action, *, step_count: int = 0):
                del step_count
                return protocols.EnvStepResult(
                    function_responses=[],
                    reward=1.0,
                    done=True,
                    is_correct=False,
                )

        results = [
            make_sample_result(text="a", candidate_id="c1"),
            make_sample_result(text="b", candidate_id="c2"),
        ]
        a = single_step_agent.SingleStepAgent(
            sampler=_FakeSampler(results),
            env_factory=lambda _: _EnvWithIsCorrect(),
            **_test_agent_kwargs(),
        )
        _, m = await a.group_rollout(
            task_data=_TestTaskData(task_id="t1"), num_samples=2
        )
        assert m["learning/sampling_accuracy"].value == pytest.approx(0.0)
        assert m["learning/pass_at_k"].value == 0.0

    async def test_thinking_level_forwarded_to_sampler_from_group_rollout(self):
        captured: list = []

        class _CapturingSampler(protocols.Sampler):
            async def sample_n(
                self,
                prompt,
                num_samples=1,
                thinking_level=None,
                **kwargs,
            ):
                captured.append(thinking_level)
                return [
                    make_sample_result(text="a", candidate_id="c1"),
                    make_sample_result(text="b", candidate_id="c2"),
                ]

        a = single_step_agent.SingleStepAgent(
            sampler=_CapturingSampler(),
            env_factory=lambda _: _FakeEnv(reward=1.0),
            **_test_agent_kwargs(),
        )
        await a.group_rollout(
            task_data=_TestTaskData(task_id="t1"), num_samples=2, thinking_level="HIGH"
        )
        assert captured == ["HIGH"]

    async def test_max_generation_tokens_forwarded_to_sampler_from_group_rollout(self):
        captured: list = []

        class _CapturingSampler(protocols.Sampler):
            async def sample_n(
                self,
                prompt,
                num_samples=1,
                max_generation_tokens=None,
                **kwargs,
            ):
                captured.append(max_generation_tokens)
                return [
                    make_sample_result(text="a", candidate_id="c1"),
                    make_sample_result(text="b", candidate_id="c2"),
                ]

        a = single_step_agent.SingleStepAgent(
            sampler=_CapturingSampler(),
            env_factory=lambda _: _FakeEnv(reward=1.0),
            **_test_agent_kwargs(),
        )
        await a.group_rollout(
            task_data=_TestTaskData(task_id="t1"),
            num_samples=2,
            max_generation_tokens=512,
        )
        assert captured == [512]

    async def test_group_rollout_default_max_generation_tokens_is_none(self):
        captured: list = []

        class _CapturingSampler(protocols.Sampler):
            async def sample_n(
                self,
                prompt,
                num_samples=1,
                max_generation_tokens=None,
                **kwargs,
            ):
                captured.append(max_generation_tokens)
                return [
                    make_sample_result(text="a", candidate_id="c1"),
                    make_sample_result(text="b", candidate_id="c2"),
                ]

        a = single_step_agent.SingleStepAgent(
            sampler=_CapturingSampler(),
            env_factory=lambda _: _FakeEnv(reward=1.0),
            **_test_agent_kwargs(),
        )
        await a.group_rollout(task_data=_TestTaskData(task_id="t1"), num_samples=2)
        assert captured == [None]


class _HangingSampler(protocols.Sampler):
    async def sample_n(
        self,
        prompt,
        num_samples=1,
        **kwargs,
    ):
        del prompt, num_samples
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class TestSingleStepAgentTimeout:
    async def test_group_rollout_marks_every_sample_as_timeout_on_fast_path_timeout(
        self,
    ):
        a = single_step_agent.SingleStepAgent(
            sampler=_HangingSampler(),
            env_factory=lambda _: _FakeEnv(),
            **_test_agent_kwargs(),
            agent_rollout_timeout_seconds=0.05,
        )
        group, m = await a.group_rollout(
            task_data=_TestTaskData(task_id="t1"), num_samples=4
        )
        assert len(group.scored_trajectories) == 4
        for g in group.scored_trajectories:
            assert math.isnan(g.reward.reward)
            assert (
                atif_extras.get_rollout_status(g.trajectory)
                == RolloutStatus.SAMPLING_FAILURE_TIMEOUT
            )
        # Keys use the sampler/trajectory_status/* prefix (one obs = one
        # trajectory). RolloutStatusMetricAggregator emits of_ratio, so
        # .value is None and .count_total is (count, total). All 4 timed out.
        assert m["sampler/trajectory_status/sampling_failure_TIMEOUT"].count_total == (
            4,
            4,
        )
        assert m["sampler/trajectory_status/sampling_failure"].count_total == (4, 4)
        assert not any(k.startswith("rollout_status/") for k in m)
        assert m["learning/sampling_accuracy"].value == pytest.approx(0.0)


# --------------------------------------------------------------------------
# TrajectoryCallback protocol + Agent plumbing (commit 1: wiring only)
# --------------------------------------------------------------------------


class _RecordingCallback:
    """Minimal in-memory TrajectoryCallback for plumbing tests.

    Records the arguments each hook is invoked with, so tests can assert
    the Agent calls them at the right points with the right payloads.
    Includes ``capture_ctx_for`` (returning None) so ``isinstance`` checks
    against the runtime-checkable TrajectoryCallback Protocol succeed.
    """

    def __init__(self) -> None:
        self.started_configs: list = []
        self.produced: list = []
        self.completed_stats: list = []
        # Records the (task_id, rollout_idx, session_id) tuple each call
        # to capture_ctx_for received. Tuple form so tests can index
        # positionally without depending on kwarg-name spelling.
        self.capture_ctx_calls: list = []
        self.trajectory_started: list = []
        self.trajectory_steps: list = []

    async def on_job_started(self, config) -> None:
        self.started_configs.append(config)

    async def on_trajectory_started(
        self,
        task_data,
        rollout_idx,
        phase="train",
        session_id=None,
        dispatch_index=None,
        traj=None,
    ) -> None:
        self.trajectory_started.append((task_data, rollout_idx, phase))

    async def on_trajectory_step_completed(
        self,
        task_data,
        rollout_idx,
        step,
        max_steps=None,
        phase="train",
        partial_steps=None,
        session_id=None,
        dispatch_index=None,
        traj=None,
    ) -> None:
        self.trajectory_steps.append((task_data, rollout_idx, step, max_steps, phase))

    async def on_trajectory_produced(
        self,
        scored,
        task_data,
        phase="train",
        rollout_idx=0,
        session_id=None,
        dispatch_index=None,
        traj=None,
    ) -> None:
        self.produced.append((scored, task_data))

    async def on_trajectory_dropped(
        self,
        task_data,
        rollout_idx,
        reason="",
        session_id=None,
        phase="train",
        dispatch_index=None,
        traj=None,
    ) -> None:
        pass

    async def on_batch_assembled(
        self, batch_id, trajectory_ids=None, train_step=None, mean_reward=None
    ) -> None:
        pass

    async def on_job_completed(self, aggregate_stats) -> None:
        self.completed_stats.append(aggregate_stats)

    async def capture_ctx_for(self, *, task_id, rollout_idx, session_id):
        """Record the resolver args and return None -- these tests care
        that the Agent calls the resolver with the right kwargs, not the
        capture layer itself."""
        self.capture_ctx_calls.append((task_id, rollout_idx, session_id))
        return None


def test_trajectory_callback_is_runtime_checkable_protocol():
    """A class implementing every hook is a TrajectoryCallback, and one
    missing a required hook is not."""
    assert isinstance(_RecordingCallback(), agent.TrajectoryCallback)

    class _Missing:
        async def on_job_started(self, config) -> None: ...

    assert not isinstance(_Missing(), agent.TrajectoryCallback)


def test_agent_rejects_callback_without_capture_ctx_for():
    """A callback missing ``capture_ctx_for`` must raise AttributeError
    so the breakage is caught immediately — not silently lose raw-LLM
    capture."""

    class _OlderCallback:
        """Only the three original TrajectoryCallback methods -- no
        ``capture_ctx_for``."""

        async def on_job_started(self, config) -> None: ...
        async def on_trajectory_produced(
            self,
            scored,
            task_data,
            phase="train",
            rollout_idx=0,
            session_id=None,
            dispatch_index=None,
            traj=None,
        ) -> None: ...
        async def on_job_completed(self, aggregate_stats) -> None: ...

    a = single_step_agent.SingleStepAgent(
        sampler=_FakeSampler([]),
        env_factory=lambda _: _FakeEnv(),
        **_test_agent_kwargs(),
        trajectory_callback=_OlderCallback(),
    )

    async def _run() -> None:
        with pytest.raises(AttributeError):
            await a._resolve_capture_ctx(
                task_id="tid", rollout_idx=0, session_id="some-session-id"
            )

    asyncio.run(_run())


def test_single_step_agent_stores_trajectory_callback():
    """SingleStepAgent accepts and stores a ``trajectory_callback`` kwarg."""
    cb = _RecordingCallback()
    a = single_step_agent.SingleStepAgent(
        sampler=None,
        env_factory=lambda _: _FakeEnv(reward=0.0),
        **_test_agent_kwargs(),
        trajectory_callback=cb,
    )
    assert a._trajectory_callback is cb


def test_single_step_agent_trajectory_callback_defaults_none():
    """Omitting ``trajectory_callback`` leaves it None (existing call-sites
    keep working, callback is opt-in)."""
    a = single_step_agent.SingleStepAgent(
        sampler=None,
        env_factory=lambda _: _FakeEnv(reward=0.0),
        **_test_agent_kwargs(),
    )
    assert a._trajectory_callback is None


@pytest.mark.asyncio
async def test_group_rollout_fires_callback_once_per_rollout():
    """Normal path: on_trajectory_produced fires once per scored trajectory,
    with the ensure_task_id-normalized task_data."""
    cb = _RecordingCallback()
    results = [make_sample_result(text=f"a{i}", candidate_id=f"c{i}") for i in range(3)]
    a = single_step_agent.SingleStepAgent(
        sampler=_FakeSampler(results),
        env_factory=lambda _: _FakeEnv(reward=0.5),
        **_test_agent_kwargs(),
        trajectory_callback=cb,
    )
    group, _ = await a.group_rollout(
        task_data=_TestTaskData(task_id="t1"), num_samples=3
    )

    assert len(cb.produced) == 3
    # Each callback got a ScoredTrajectory and the (task_id-populated) task_data.
    for scored, td in cb.produced:
        assert scored.trajectory.session_id is not None
        assert td.task_id == "t1"
    # Same scored objects the group returned.
    produced_sids = {g.trajectory.session_id for g, _ in cb.produced}
    group_sids = {g.trajectory.session_id for g in group.scored_trajectories}
    assert produced_sids == group_sids


@pytest.mark.asyncio
async def test_group_rollout_fires_callback_for_every_timeout_trajectory():
    """Timeout fast-path: on_trajectory_produced still fires for each
    timeout trajectory (spec §4.2 write-every-rollout)."""
    cb = _RecordingCallback()
    a = single_step_agent.SingleStepAgent(
        sampler=_HangingSampler(),
        env_factory=lambda _: _FakeEnv(),
        **_test_agent_kwargs(),
        agent_rollout_timeout_seconds=0.05,
        trajectory_callback=cb,
    )
    group, _ = await a.group_rollout(
        task_data=_TestTaskData(task_id="t1"), num_samples=4
    )

    assert len(group.scored_trajectories) == 4
    assert len(cb.produced) == 4
    for scored, _td in cb.produced:
        assert math.isnan(scored.reward.reward)


@pytest.mark.asyncio
async def test_group_rollout_swallows_callback_exception():
    """A callback that raises must not crash the rollout; the group is
    still returned complete."""

    class _BoomCallback(_RecordingCallback):
        async def on_trajectory_produced(
            self,
            scored,
            task_data,
            phase="train",
            rollout_idx=0,
            session_id=None,
            dispatch_index=None,
            traj=None,
        ) -> None:
            raise RuntimeError("boom")

    results = [make_sample_result(text=f"a{i}", candidate_id=f"c{i}") for i in range(2)]
    a = single_step_agent.SingleStepAgent(
        sampler=_FakeSampler(results),
        env_factory=lambda _: _FakeEnv(reward=0.5),
        **_test_agent_kwargs(),
        trajectory_callback=_BoomCallback(),
    )
    group, _ = await a.group_rollout(
        task_data=_TestTaskData(task_id="t1"), num_samples=2
    )
    assert len(group.scored_trajectories) == 2  # rollout unharmed


@pytest.mark.asyncio
async def test_rollout_without_callback_does_not_fire():
    """No callback configured -> no attempt to invoke one (regression guard
    for the None path)."""
    results = [make_sample_result(text="a0", candidate_id="c0")]
    a = single_step_agent.SingleStepAgent(
        sampler=_FakeSampler(results),
        env_factory=lambda _: _FakeEnv(reward=0.5),
        **_test_agent_kwargs(),
    )
    group, _ = await a.group_rollout(
        task_data=_TestTaskData(task_id="t1"), num_samples=1
    )
    assert len(group.scored_trajectories) == 1


class _BoomError(RuntimeError):
    """Distinctive error injected into exactly one rollout of a group."""


@pytest.mark.asyncio
async def test_group_rollout_publishes_survivors_when_one_rollout_raises():
    """A mid-group rollout failure must not cancel its in-flight siblings
    before they publish. The group still raises (so the caller drops the
    whole group -- contract preserved), but every surviving sibling must
    have fired on_trajectory_produced (write-every-rollout, spec §4.2).

    Injection seam: an env_factory that raises on exactly one call. Because
    SingleStepAgent's per-rollout _run calls env_factory before publishing,
    this reproduces the "sibling cancelled mid-publish" bug precisely -- the
    bare gather cancels the other _run coroutines the moment this one raises.
    """
    published: list = []
    num_samples = 4
    fail_idx = 1

    class _SlowPublishCb(_RecordingCallback):
        async def on_trajectory_produced(
            self,
            scored,
            task_data,
            phase="train",
            rollout_idx=0,
            session_id=None,
            dispatch_index=None,
            traj=None,
        ) -> None:
            # Publishing takes a beat (mimics the real disk/GCS write). This
            # widens the window in which a bare gather would cancel this
            # sibling mid-publish when another rollout raises first.
            await asyncio.sleep(0.05)
            published.append(scored.trajectory.session_id)

    class _RaisingEnv(_FakeEnv):
        """Env whose async step raises AFTER a delay -- so its sibling
        rollouts have already advanced into their own publish await when the
        failure fires (reproducing sibling-cancelled-mid-publish)."""

        async def step(self, action: protocols.Action, *, step_count: int = 0):
            del action, step_count
            await asyncio.sleep(0.01)
            raise _BoomError("env.step boom")

    call_count = {"n": 0}

    def _env_factory(_task_data):
        i = call_count["n"]
        call_count["n"] += 1
        return _RaisingEnv() if i == fail_idx else _FakeEnv(reward=0.5)

    a = single_step_agent.SingleStepAgent(
        sampler=_FakeSampler(
            [
                make_sample_result(text=f"a{i}", candidate_id=f"c{i}")
                for i in range(num_samples)
            ]
        ),
        env_factory=_env_factory,
        **_test_agent_kwargs(),
        trajectory_callback=_SlowPublishCb(),
    )

    with pytest.raises(_BoomError):
        await a.group_rollout(
            task_data=_TestTaskData(task_id="t1"), num_samples=num_samples
        )

    # The three non-failing rollouts must have published despite the sibling
    # failure (they were NOT cancelled mid-publish).
    assert len(published) == num_samples - 1, published


# --------------------------------------------------------------------------
# capture_ctxs forwarding: SingleStepAgent.rollout / group_rollout -> sampler.sample
# --------------------------------------------------------------------------


class _CapturingSampler(protocols.Sampler):
    """Sampler that records the ``capture_ctxs`` kwarg every ``sample`` call
    received, so tests can assert the Agent forwarded the resolved contexts.

    ``capture_ctxs_calls`` is a list-of-lists: each entry is the
    per-sample-call ``list[CaptureContext | None]`` (post per-candidate
    reshape). Single-rollout tests expect ``[[ctx]]``; batched-fanout
    tests expect ``[[ctx0, ctx1, ...]]``.
    """

    def __init__(self, results):
        self._results = results
        self.capture_ctxs_calls: list = []

    async def sample_n(
        self,
        prompt,
        num_samples=1,
        capture_ctxs=None,
        **kwargs,
    ):
        del prompt
        # Preserve None-vs-list distinction and record a shallow copy so
        # sibling mutations (unlikely) can't corrupt earlier records.
        self.capture_ctxs_calls.append(
            list(capture_ctxs) if capture_ctxs is not None else None
        )
        return list(self._results[:num_samples])


class _CtxReturningCallback(_RecordingCallback):
    """TrajectoryCallback whose ``capture_ctx_for`` returns a canned sentinel
    (so tests can assert forwarding identity)."""

    def __init__(self, ctx) -> None:
        super().__init__()
        self._ctx = ctx

    async def capture_ctx_for(self, *, task_id, rollout_idx, session_id):
        self.capture_ctx_calls.append((task_id, rollout_idx, session_id))
        return self._ctx


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_rollout_with_callback_missing_capture_ctx_for_raises():
    """A callback missing ``capture_ctx_for`` must raise AttributeError
    during rollout — fail loud so the contract violation is caught."""

    class _OlderCallback:
        async def on_job_started(self, config) -> None: ...
        async def on_trajectory_produced(
            self,
            scored,
            task_data,
            phase="train",
            rollout_idx=0,
            session_id=None,
            dispatch_index=None,
            traj=None,
        ) -> None: ...
        async def on_job_completed(self, aggregate_stats) -> None: ...

    sampler = _CapturingSampler([make_sample_result(text="a0", candidate_id="c0")])
    a = single_step_agent.SingleStepAgent(
        sampler=sampler,
        env_factory=lambda _: _FakeEnv(reward=0.5),
        **_test_agent_kwargs(),
        trajectory_callback=_OlderCallback(),
    )
    with pytest.raises(AttributeError):
        await a.rollout(_TestTaskData(task_id="t1"))


# --------------------------------------------------------------------------
# group_rollout: N per-rollout capture_ctxs, passed as a list of length N.
# Closes the batched-fanout gap identified by spec review — the
# batched sampler wire call now populates the raw-LLM ledger for word_puzzle
# (which always goes through group_rollout via RolloutPipeline.rollout_task).
# --------------------------------------------------------------------------


class _MultiCtxReturningCallback(_RecordingCallback):
    """TrajectoryCallback whose ``capture_ctx_for(task_id, rollout_idx,
    session_id)`` returns a distinct CaptureContext per session_id, so
    batched-fanout tests can assert the N-ctx resolution + ordering."""

    def __init__(self) -> None:
        super().__init__()
        self._resolved: dict = {}

    async def capture_ctx_for(self, *, task_id, rollout_idx, session_id):
        from pathlib import Path as _P

        from utils.raw_capture import CaptureContext, Counter

        self.capture_ctx_calls.append((task_id, rollout_idx, session_id))
        ctx = self._resolved.get(session_id)
        if ctx is None:
            ctx = CaptureContext(
                trial_dir=_P(f"/tmp/nonexistent/{session_id}"),
                trajectory_id=session_id,
                counter=Counter(),
            )
            self._resolved[session_id] = ctx
        return ctx


@pytest.mark.asyncio
async def test_group_rollout_resolves_N_per_rollout_capture_ctxs_and_passes_as_list() -> (
    None
):
    """batched-fanout path: sampler receives ``capture_ctxs = [ctx0, ctx1, ctx2]``
    where each ctx corresponds to ``session_id_for(rollout_idx=i)``.

    Verifies the ORDER is stable (ctx[i] matches rollout i's session_id) and
    that the resolver is called num_samples times, once per rollout_idx,
    with (task_id, rollout_idx, session_id) kwargs.
    """
    num_samples = 3
    cb = _MultiCtxReturningCallback()
    sampler = _CapturingSampler(
        [
            make_sample_result(text=f"a{i}", candidate_id=f"c{i}")
            for i in range(num_samples)
        ]
    )
    a = single_step_agent.SingleStepAgent(
        sampler=sampler,
        env_factory=lambda _: _FakeEnv(reward=0.5),
        **_test_agent_kwargs(),
        trajectory_callback=cb,
    )
    await a.group_rollout(_TestTaskData(task_id="t1"), num_samples=num_samples)

    # Exactly one sample() call (batched fanout).
    assert len(sampler.capture_ctxs_calls) == 1
    forwarded = sampler.capture_ctxs_calls[0]
    assert forwarded is not None
    assert len(forwarded) == num_samples
    # Each ctx matches the resolved-per-session_id ctx.
    assert len(cb.capture_ctx_calls) == num_samples
    for i in range(num_samples):
        call_task_id, call_rollout_idx, sid_i = cb.capture_ctx_calls[i]
        assert call_task_id == "t1"
        assert call_rollout_idx == i
        assert forwarded[i] is cb._resolved[sid_i]
    # Resolved session_ids encode rollout_idx and appear in order (0, 1, 2).
    for i, (_tid, _ridx, sid) in enumerate(cb.capture_ctx_calls):
        assert f"-{i}-" in sid, f"expected rollout_idx {i} in session_id {sid!r}"


@pytest.mark.asyncio
async def test_group_rollout_without_callback_passes_all_none_capture_ctxs_list() -> (
    None
):
    """No trajectory callback -> group_rollout passes ``capture_ctxs=[None]*N``
    (consistent shape with the with-callback path; sampler treats element-None
    as a no-op skip for that candidate)."""
    num_samples = 3
    sampler = _CapturingSampler(
        [
            make_sample_result(text=f"a{i}", candidate_id=f"c{i}")
            for i in range(num_samples)
        ]
    )
    a = single_step_agent.SingleStepAgent(
        sampler=sampler,
        env_factory=lambda _: _FakeEnv(reward=0.5),
        **_test_agent_kwargs(),
    )
    await a.group_rollout(_TestTaskData(task_id="t1"), num_samples=num_samples)
    assert sampler.capture_ctxs_calls == [[None, None, None]]


class _IncludeThoughtsEnv(_FakeEnv):
    def get_prompt(self):
        from google.cloud.aiplatform_v1.types import GenerationConfig

        prompt = _prompt_gcr(self._prompt)
        prompt.generation_config = GenerationConfig(
            thinking_config=GenerationConfig.ThinkingConfig(include_thoughts=True)
        )
        return prompt


class TestAgentExtraGenerationParams:
    """``Trajectory.agent.extra`` records the rollout's generation knobs."""

    async def test_group_rollout_records_knobs_and_prompt_include_thoughts(self):
        a = single_step_agent.SingleStepAgent(
            sampler=_FakeSampler([make_sample_result(text="a", candidate_id="c1")]),
            env_factory=lambda _: _IncludeThoughtsEnv(),
            **_test_agent_kwargs(),
        )
        group, _ = await a.group_rollout(
            _TestTaskData(task_id="t1"),
            num_samples=1,
            thinking_level="HIGH",
            max_generation_tokens=1024,
        )
        assert group.scored_trajectories[0].trajectory.agent.extra == {
            "thinking_level": "HIGH",
            "max_generation_tokens": 1024,
            "include_thoughts": True,
        }

    async def test_agent_extra_is_none_without_knobs(self):
        a = single_step_agent.SingleStepAgent(
            sampler=_FakeSampler([make_sample_result(text="a", candidate_id="c1")]),
            env_factory=lambda _: _FakeEnv(),
            **_test_agent_kwargs(),
        )
        group, _ = await a.group_rollout(_TestTaskData(task_id="t1"), num_samples=1)
        assert group.scored_trajectories[0].trajectory.agent.extra is None

    async def test_base_rollout_records_knobs(self):
        class _EchoAgent(agent.Agent):
            async def _rollout_internal(
                self, task_data: TaskData, ctx: agent.RolloutContext
            ) -> ScoredTrajectory:
                return ScoredTrajectory(trajectory=ctx.traj, reward=Reward(reward=1.0))

        a = _EchoAgent(
            sampler=_FakeSampler([]),
            env_factory=lambda _: _FakeEnv(),
            recipe_name="test",
            tuning_job_id="tj-test",
        )
        scored = await a.rollout(_TestTaskData(task_id="t1"), thinking_level="LOW")
        assert scored is not None
        assert scored.trajectory.agent.extra == {"thinking_level": "LOW"}
