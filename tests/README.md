# Tests

```
conftest.py             # repo-root: sys.path + env-var reset fixtures
pyproject.toml          # pytest + coverage config
tests/
  unit/                 # fast, per-module unit tests (~90 files)
    fixtures/           # recorded + synthetic wire payloads
  integration/          # full-cycle tests with the transport mocked
```

`tests/unit/` mirrors the source tree: for `src/<pkg>/foo.py` the test is
`tests/unit/test_foo.py`. The set changes often, so `ls tests/unit` is the
authoritative list rather than a table here.

`pyproject.toml` configures pytest-asyncio in `auto` mode; async test
functions/methods do not need an explicit `@pytest.mark.asyncio` decorator.

The `integration` marker is registered but currently unused —
`tests/integration/test_atif_end_to_end.py` is collected via `testpaths`
without it.

## Setup

```sh
uv sync --extra test
```

Add `--extra agent_solver` for the `minisweagent`-backed word_puzzle solver
tests; without it they `importorskip` cleanly.

## Running

```sh
uv run --extra test python -m pytest                                # all (~40s)
uv run --extra test python -m pytest tests/unit -v
uv run --extra test python -m pytest --cov --cov-report=term-missing --cov-report=html

# one module
uv run --extra test python -m pytest tests/unit/test_vertex_msrl_backend.py -v
# one test
uv run --extra test python -m pytest \
  tests/unit/test_async_rl_train_loop.py::TestTrainAndEvalLoopAsync::test_runs_one_iteration_and_calls_train -v
```

## Adding tests

  - For `foo.py`, add `tests/unit/test_foo.py`. For `subpkg/foo.py`, mirror
    the layout under `tests/unit/subpkg/test_foo.py`.
  - Use `IsolatedAsyncioTestCase` or plain `async def` (auto mode handles both).
  - For training-loop tests with fake components, follow the patterns in
    `tests/unit/test_async_rl_train_loop.py` and
    `tests/unit/test_train_and_eval_loop.py`.
  - For backend tests, see `tests/unit/test_vertex_msrl_backend.py`. The
    `vertex_msrl_rest` backend is REST over `httpx` with a two-phase LRO
    (create POST, then poll); the tests stub both phases on
    `VertexMsrlRestClient` with `MagicMock`/`AsyncMock` rather than hitting
    the network.
  - Prefer naming a symbol over citing `file.py:<line>` — line numbers in
    comments go stale silently and have had to be repaired en masse before.
