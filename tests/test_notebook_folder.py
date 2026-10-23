"""The walkthrough notebook is self-contained and shows exactly the code that lives in src/."""
import json

import pytest

from scripts.build_notebook import DATA_COPY, DATA_SOURCE, EMBEDDED, NOTEBOOK, segment

REBUILD = "run: python scripts/build_notebook.py --execute"


def notebook_code():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def test_notebook_imports_nothing_from_the_project():
    prefixes = ("from src", "import src", "from edupredict", "import edupredict")
    assert [line for line in notebook_code().splitlines() if line.startswith(prefixes)] == []


@pytest.mark.parametrize("module,name", [(m, n) for m, groups in EMBEDDED.items() for g in groups for n in g])
def test_notebook_shows_the_same_code_as_src(module, name):
    assert segment(module, name) in notebook_code(), f"{name} differs from src/{module}.py - {REBUILD}"


def test_notebook_dataset_copy_matches_source():
    assert DATA_COPY.read_bytes() == DATA_SOURCE.read_bytes(), f"dataset copy is out of date - {REBUILD}"
