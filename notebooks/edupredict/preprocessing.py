"""Loading, encoding, scaling and splitting of the synthetic EduPredict dataset.

Everything fitted (means, stds, category lists) comes from the TRAIN split only
and is stored in a JSON config so exactly the same transform can be applied to new rows later.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA_PATH = "data/raw/edupredict_synthetic.csv"
TARGET = "final_marks"  # already on a 0-100 scale

# Input columns by when they become available in a term. Variants add groups cumulatively.
EARLY_NUMERIC = ["age", "grade_level", "parent_education", "family_income_band", "commute_minutes",
                 "extracurricular_hours", "sleep_hours", "stress_level", "health_status", "past_failures",
                 "subject_interest", "class_size", "prev_year_score"]
TERM_NUMERIC = ["study_hours_week", "attendance_pct", "assignment_completion_pct", "late_submissions",
                "class_participation", "quiz_avg_score"]
LATE_NUMERIC = ["midterm_score", "assignment_avg_score"]
BINARY = ["gender", "internet_access", "part_time_job", "tutoring"]
CATEGORICAL = ["subject"]
VARIANT_NUMERIC = {
    "early": EARLY_NUMERIC,
    "during_term": EARLY_NUMERIC + TERM_NUMERIC,
    "late": EARLY_NUMERIC + TERM_NUMERIC + LATE_NUMERIC,
}
# Never inputs: contain or derive from the target.
EXCLUDED = ["final_exam_score", "grade_letter", "passed", TARGET]


def load_raw(path=DATA_PATH):
    """Read the synthetic dataset from a CSV file (create it with the synthetic_data module if it is missing)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found - generate it with the synthetic_data module first")
    df = pd.read_csv(path)
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values")
    return df


def group_split(df, seed=42, val=0.15, test=0.15):
    """70/15/15 split in which every student (all 6 of their rows) stays in one split."""
    groups = df["student_id"]
    outer = GroupShuffleSplit(n_splits=1, test_size=test, random_state=seed)
    trainval, test_idx = next(outer.split(df, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=val / (1 - test), random_state=seed)
    tr, va = next(inner.split(df.iloc[trainval], groups=groups.iloc[trainval]))
    return trainval[tr], trainval[va], test_idx


def fit_config(train_df, variant="late"):
    """Learn scaling / category lists from the training split only."""
    features = []
    for col in VARIANT_NUMERIC[variant]:
        s = train_df[col]
        features.append({"name": col, "kind": "numeric", "mean": float(s.mean()),
                         "std": float(s.std(ddof=0)) or 1.0, "min": float(s.min()),
                         "max": float(s.max()), "default": float(s.median()),
                         "integer": bool((s == s.round()).all())})
    for col in BINARY:
        opts = sorted(train_df[col].unique().tolist())
        features.append({"name": col, "kind": "binary", "options": opts,
                         "default": train_df[col].mode().tolist()[0]})
    for col in CATEGORICAL:
        opts = sorted(train_df[col].unique().tolist())
        features.append({"name": col, "kind": "categorical", "options": opts,
                         "default": train_df[col].mode().tolist()[0]})
    return {"features": features, "target_mean": float(train_df[TARGET].mean()),
            "target_std": float(train_df[TARGET].std(ddof=0)), "variant": variant}


def column_names(cfg):
    """Names of the network input columns, in order."""
    names = []
    for f in cfg["features"]:
        if f["kind"] == "categorical":
            names += [f"{f['name']}={o}" for o in f["options"]]
        else:
            names.append(f["name"])
    return names


def transform(df, cfg):
    """Raw rows -> scaled float matrix (n, n_inputs). Rejects unseen categories."""
    cols = []
    for f in cfg["features"]:
        v = df[f["name"]]
        if f["kind"] == "numeric":
            cols.append(((v - f["mean"]) / f["std"]).to_numpy(float))
            continue
        if not v.isin(f["options"]).all():
            raise ValueError(f"{f['name']}: value not in {f['options']}")
        if f["kind"] == "binary":
            cols.append((v == f["options"][1]).to_numpy(float))
        else:
            cols += [(v == o).to_numpy(float) for o in f["options"]]
    return np.column_stack(cols)


def scale_target(marks, cfg):
    return (np.asarray(marks, float) - cfg["target_mean"]) / cfg["target_std"]


def unscale_target(z, cfg):
    return np.asarray(z, float) * cfg["target_std"] + cfg["target_mean"]


def grade_letter(marks):
    """Letter band used by the dataset (A>=85, B>=70, C>=55, D>=40, else F)."""
    return next(g for g, lo in (("A", 85), ("B", 70), ("C", 55), ("D", 40), ("F", -1e9)) if marks >= lo)


def save_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
