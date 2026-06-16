"""pytest bootstrap so the test suite can import the package without an explicit install.

The package lives at ``src/ebrag``. ``pyproject.toml`` already configures the src layout
for ``pip install -e .``, but this conftest also lets a reviewer run ``pytest -q`` from
a freshly-extracted artefact zip without the install step.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
