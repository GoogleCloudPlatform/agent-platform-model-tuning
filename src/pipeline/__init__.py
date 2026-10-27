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

"""pipeline — RL rollout pipelines + trainer/eval entry points.

Nothing is re-exported at the package level; import submodules directly:

    from pipeline.sync_rollout_pipeline import SyncRolloutPipeline
    from pipeline.train_and_eval import train
    from pipeline.eval_only import EvalPipeline

The shared domain model (``Agent``, ``JsonlDataset``, ``Trajectory``,
protocols, metrics, aggregators, recipe ``Config`` bases, reward
normalizer registry, ...) lives in the top-level ``core`` package.
Import those from ``core.*`` directly.
"""
