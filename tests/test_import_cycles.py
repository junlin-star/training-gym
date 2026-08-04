"""Guard against import cycles between ``common`` and ``cli``.

``cli`` imports ``common.config``, so anything ``common.config`` needs from
``cli`` must be imported lazily inside the function that uses it. A module-level
import there cycles: ``common.config`` stops halfway, ``cli`` re-enters it, and
the partially initialized module has no ``get_dashboard_url`` yet.

Each module is imported *first* in a fresh interpreter. That ordering is the
whole point — inside pytest these modules are already in ``sys.modules``, which
masks the cycle no matter which side is broken.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


CYCLE_MODULES = [
    "modal_training_gym.common.config",
    "modal_training_gym.cli",
    "modal_training_gym.cli.client",
    "modal_training_gym.cli.setup",
]


@pytest.mark.parametrize("module", CYCLE_MODULES)
def test_module_imports_first_in_a_fresh_interpreter(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"importing {module} first fails:\n{result.stderr}"
