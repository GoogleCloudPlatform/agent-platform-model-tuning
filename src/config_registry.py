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

"""Two-level config registry: recipe namespace + variant.

Config names use ``recipe/variant`` format (``rpartition("/")`` split):

- ``word_puzzle/async``, ``word_puzzle/sync``

Each recipe maps to a module with a ``CONFIGS: dict[str, RunConfig]`` at
module scope. ``get_config(name)`` lazily imports the module and looks up
the variant. No import-time side effects, no ``register_all()`` dance.

Additional registries may extend the shared ``_RECIPES`` dict with a
namespace prefix to avoid collisions with the default namespaces.
"""

from __future__ import annotations

import importlib
from typing import Any

# recipe-namespace -> dotted module path containing a CONFIGS dict.
# Additional registries may extend this same dict with a namespace prefix.
_RECIPES: dict[str, str] = {
    "word_puzzle": "example_recipe.word_puzzle.single_step.configs",
    "word_puzzle_agent": (
        "example_recipe.word_puzzle.agent_solver.minisweagent.agent_solver_config"
    ),
    "word_puzzle_simple": (
        "example_recipe.word_puzzle.agent_solver.simple_agent.config_model"
    ),
}


def get_config(name: str) -> Any:
    """Resolve ``recipe/variant`` to a RunConfig.

    Split on the LAST ``/``: everything before is the recipe namespace,
    everything after is the variant key within that recipe's CONFIGS dict.
    """
    recipe, sep, variant = name.rpartition("/")
    if not sep:
        raise KeyError(f"config name must be 'recipe/variant', got {name!r}")
    if recipe not in _RECIPES:
        known = ", ".join(sorted(_RECIPES))
        raise KeyError(f"unknown recipe {recipe!r}; known: {known}")
    module = importlib.import_module(_RECIPES[recipe])
    try:
        return module.CONFIGS[variant]
    except KeyError:
        known = ", ".join(sorted(module.CONFIGS))
        raise KeyError(
            f"unknown variant {variant!r} for {recipe!r}; known: {known}"
        ) from None


def recipe_for(name: str) -> str:
    """Return the recipe namespace from a ``recipe/variant`` config name.

    This is the raw namespace key used in ``_RECIPES`` (e.g.
    ``"word_puzzle"``, or a prefixed key such as ``"<ns>/<recipe>"``).
    For ATIF trajectory tagging use :func:`recipe_label` instead — it
    strips any namespace prefix so paths and session IDs match the short
    recipe identity.
    """
    recipe, sep, _ = name.rpartition("/")
    if not sep:
        raise KeyError(f"config name must be 'recipe/variant', got {name!r}")
    return recipe


def recipe_label(name: str) -> str:
    """Return the short recipe label for ATIF / session-id tagging.

    Strips the ``internal/`` namespace prefix that a prefixed registry
    may add, so ``"internal/<recipe>/async"`` → ``"<recipe>"``.  Default
    configs pass through unchanged: ``"word_puzzle/async"`` →
    ``"word_puzzle"``.
    """
    ns = recipe_for(name)
    return ns.removeprefix("internal/")
