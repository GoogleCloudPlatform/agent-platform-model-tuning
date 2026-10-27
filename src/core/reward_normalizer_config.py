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

"""RewardNormalizerConfig: abstract selector for reward normalization.

A concrete subclass IS the normalizer choice; its ``make()`` builds the live
``RewardNormalizer`` (the behavior ABC in ``core.protocols``). This parallels
``BackendConfig`` (``backends/backend_config.py``): the config is frozen data,
``make()`` returns the runtime object. Replaces the older
``get_reward_normalizer(name)`` + ``REWARD_NORMALIZERS`` registry dispatch.

The concrete configs carry no fields today (the normalizers are zero-arg); they
are the natural home for any future per-normalizer hyperparameter without
reintroducing a string registry.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from core.protocols import RewardNormalizer


@dataclass(frozen=True)
class RewardNormalizerConfig(abc.ABC):
    """Abstract base: a concrete subclass selects the normalizer; ``make()``
    builds it."""

    @abc.abstractmethod
    def make(self) -> RewardNormalizer: ...


@dataclass(frozen=True)
class CenteringNormalizerConfig(RewardNormalizerConfig):
    """Selects the per-prompt mean-centering normalizer."""

    def make(self) -> RewardNormalizer:
        # Lazy import keeps this src/ config module decoupled from the behavior
        # module at import time (both import core.protocols).
        from core.reward_normalizer import CenteringNormalizer

        return CenteringNormalizer()


@dataclass(frozen=True)
class GRPONormalizerConfig(RewardNormalizerConfig):
    """Selects the GRPO-style per-prompt standardizing normalizer."""

    def make(self) -> RewardNormalizer:
        from core.reward_normalizer import GRPONormalizer

        return GRPONormalizer()
