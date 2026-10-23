"""Refresh the self-contained notebooks/ folder from the project sources.

notebooks/ is meant to be copied on its own, so it carries snapshot copies of the code (src/*.py) and the dataset.
Run `python scripts/sync_notebook_folder.py` after changing src/ or the data;
tests/test_notebook_folder.py fails if the copies drift.
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = ["ann.py", "preprocessing.py", "evaluation.py", "learning_rules.py", "synthetic_data.py"]
COPIES = [(ROOT / "src" / name, ROOT / "notebooks" / "edupredict" / name) for name in CODE_FILES] + [
    (ROOT / "data" / "raw" / "edupredict_synthetic.csv", ROOT / "notebooks" / "data" / "edupredict_synthetic.csv")]
INIT = '"""Helper code for the EduPredict AI walkthrough notebook."""\n'


def sync():
    for source, target in COPIES:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (ROOT / "notebooks" / "edupredict" / "__init__.py").write_text(INIT, encoding="utf-8")
    return [target for _, target in COPIES]


if __name__ == "__main__":
    for target in sync():
        print("synced", target.relative_to(ROOT))
