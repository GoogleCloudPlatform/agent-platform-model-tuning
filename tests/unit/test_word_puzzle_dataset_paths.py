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

"""Default dataset paths for the word_puzzle recipe must be robust to CWD.

The defaults point into the recipe's ``dataset/`` sub-package
(``src/example_recipe/word_puzzle/dataset/combined_*_word_transformation_v1.jsonl``).
Historically they were bare relative filenames, so a run launched from any
directory other than the recipe dir crashed at dataset load with
``FileNotFoundError``. These tests pin that the defaults resolve to an
absolute path under ``dataset/`` no matter what the process CWD is, while
leaving caller-supplied paths untouched.

The ``.jsonl`` files themselves are not in every checkout — distributions
that ship only ``dataset/puzzle_dataset_gen.py`` and
``dataset/gen_withhold_eval.py`` are expected, so the existence checks
skip when the data is absent instead of failing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from example_recipe.word_puzzle.config_model import WordPuzzleTaskConfig


@pytest.fixture
def chdir_tmp(tmp_path, monkeypatch):
    """Run the body with CWD set to an unrelated temp dir.

    This is the exact condition that broke real runs: launching from a
    sandbox dir that does not contain the dataset files.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_default_train_path_is_absolute_regardless_of_cwd(chdir_tmp):
    p = WordPuzzleTaskConfig().train_data_path
    assert os.path.isabs(p), f"default train_data_path is CWD-relative: {p!r}"


def test_default_eval_path_is_absolute_regardless_of_cwd(chdir_tmp):
    p = WordPuzzleTaskConfig().eval_data_path
    assert os.path.isabs(p), f"default eval_data_path is CWD-relative: {p!r}"


def test_default_paths_load_when_the_data_ships(chdir_tmp):
    """Where the generated splits are checked in, the defaults must find them."""
    cfg = WordPuzzleTaskConfig()
    for label, p in (("train", cfg.train_data_path), ("eval", cfg.eval_data_path)):
        if not os.path.exists(p):
            pytest.skip(f"{label} split not shipped in this checkout: {p}")
        assert os.path.isfile(p), f"default {label}_data_path is not a file: {p!r}"


def test_default_paths_point_into_dataset_dir():
    """The resolved defaults should live in the recipe's dataset/ sub-package."""
    dataset_dir = Path(WordPuzzleTaskConfig().train_data_path).resolve().parent
    # The dataset sub-package lives one level below the recipe package dir.
    assert dataset_dir.name == "dataset", dataset_dir
    assert (dataset_dir / "__init__.py").is_file(), dataset_dir


def test_caller_supplied_path_is_not_rewritten():
    """An explicit path (absolute or relative) must be preserved verbatim."""
    custom = "/some/explicit/train.jsonl"
    assert WordPuzzleTaskConfig(train_data_path=custom).train_data_path == custom
    assert WordPuzzleTaskConfig(eval_data_path="rel/eval.jsonl").eval_data_path == (
        "rel/eval.jsonl"
    )
