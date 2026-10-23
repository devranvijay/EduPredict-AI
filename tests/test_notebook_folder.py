"""notebooks/ is a self-contained folder: its copies of the code and the dataset must match the project sources."""
import json
from pathlib import Path

import pytest

from scripts.sync_notebook_folder import COPIES

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "edupredict_walkthrough.ipynb"


@pytest.mark.parametrize("source,target", COPIES, ids=[target.name for _, target in COPIES])
def test_notebook_folder_copy_matches_source(source, target):
    assert target.exists(), "missing copy - run: python scripts/sync_notebook_folder.py"
    assert target.read_bytes() == source.read_bytes(), f"{target.name} is out of date - run: python scripts/sync_notebook_folder.py"


def test_notebook_does_not_depend_on_project_sources():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    assert "from src" not in code and "import src" not in code
    assert "from edupredict" in code
