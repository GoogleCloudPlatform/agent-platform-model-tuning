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

"""Offline analysis of published ATIF trajectory trees.

Loads a job's trial tree (local or ``gs://``) and computes metrics over
it -- either for one run, or side-by-side across two runs. Metric
aggregation reuses ``core.metric_aggregators`` / ``core.metric_spec``, so
offline numbers match what the eval pipeline reported for the same run.
"""

from .loader import TrialLoadError, TrialRef, TrialView, discover_trials, load_trial
from .metrics import SXS_METRICS, register_sxs_metric
from .runner import (
    AnalysisResult,
    aggregate_pairs,
    aggregate_views,
    analyze_run,
    analyze_sxs,
    select_refs,
)
from .tree_reader import TreeReadError

__all__ = [
    "SXS_METRICS",
    "AnalysisResult",
    "TreeReadError",
    "TrialLoadError",
    "TrialRef",
    "TrialView",
    "aggregate_pairs",
    "aggregate_views",
    "analyze_run",
    "analyze_sxs",
    "discover_trials",
    "load_trial",
    "register_sxs_metric",
    "select_refs",
]
