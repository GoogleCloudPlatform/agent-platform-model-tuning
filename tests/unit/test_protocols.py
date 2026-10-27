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

"""Test Sampler ABC signature contract, including capture_ctxs kwarg on sample_n."""

import inspect

from core.protocols import Sampler


class TestSamplerSampleNSignature:
    """Verify the abstract ``sample_n`` method's capture_ctxs kwarg.

    ``capture_ctxs`` lives on ``sample_n`` (the abstract N-sample method),
    NOT on the convenience ``sample`` (which delegates to ``sample_n``
    with ``num_samples=1`` and accepts a singular ``capture_ctx``).
    """

    def test_sample_n_accepts_capture_ctxs_kwarg(self) -> None:
        sig = inspect.signature(Sampler.sample_n)
        assert "capture_ctxs" in sig.parameters, (
            f"Sampler.sample_n missing capture_ctxs kwarg; has {list(sig.parameters)}"
        )

    def test_capture_ctxs_defaults_to_none(self) -> None:
        sig = inspect.signature(Sampler.sample_n)
        param = sig.parameters["capture_ctxs"]
        assert param.default is None

    def test_capture_ctxs_is_keyword_or_keyword_only(self) -> None:
        # Any positional/keyword position is fine, but the default must
        # be None so existing callers don't have to pass it.
        sig = inspect.signature(Sampler.sample_n)
        param = sig.parameters["capture_ctxs"]
        assert param.default is None

    def test_capture_ctxs_annotation_is_sequence_of_optional(self) -> None:
        """capture_ctxs is annotated as ``Sequence[CaptureContext | None] | None``.

        Inspects the raw string annotation (forward-ref form) rather than
        resolving via ``get_type_hints`` -- ``CaptureContext`` lives behind
        a TYPE_CHECKING seam to avoid a runtime import cycle, so eager
        resolution would fail. Guards against accidental regression to a
        single-ctx kwarg (``capture_ctx: CaptureContext | None``).
        """
        raw = Sampler.sample_n.__annotations__.get("capture_ctxs", "")
        assert "Sequence" in raw, (
            f"expected 'Sequence' in capture_ctxs annotation, got {raw!r}"
        )
