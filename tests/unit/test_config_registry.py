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

"""Two-level config registry.

``config_registry.get_config("word_puzzle/async")`` lazily imports the recipe
module and returns the matching ``RunConfig``. ``recipe_for`` extracts
the recipe namespace from the ``recipe/variant`` config name.
"""

from __future__ import annotations

from core.run_config import RunConfig
from config_registry import get_config, recipe_for, recipe_label


def test_get_returns_run_config():
    assert isinstance(get_config("word_puzzle/async"), RunConfig)
    assert isinstance(get_config("word_puzzle/sync"), RunConfig)


def test_recipe_for_plain_names():
    assert recipe_for("word_puzzle/async") == "word_puzzle"
    assert recipe_for("word_puzzle/sync") == "word_puzzle"


def test_get_unknown_recipe_raises():
    import pytest

    with pytest.raises(KeyError, match="unknown recipe"):
        get_config("nonexistent/async")


def test_get_unknown_variant_raises():
    import pytest

    with pytest.raises(KeyError, match="unknown variant"):
        get_config("word_puzzle/nonexistent_variant")


def test_recipe_for_no_slash_raises():
    import pytest

    with pytest.raises(KeyError, match="recipe/variant"):
        recipe_for("word_puzzle_async")


def test_recipe_label_plain_names():
    """recipe_label strips nothing from an unprefixed config name."""
    assert recipe_label("word_puzzle/async") == "word_puzzle"
    assert recipe_label("word_puzzle/sync") == "word_puzzle"
