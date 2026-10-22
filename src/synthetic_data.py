"""Generate a SYNTHETIC student-performance dataset (one row per student x subject).

All values are simulated from the causal story written below - not real students.
Run: python -m src.synthetic_data  ->  data/raw/edupredict_synthetic.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
N_STUDENTS = 2000
SUBJECTS_PER_STUDENT = 6
# subject -> difficulty (higher = harder, lowers marks)
SUBJECTS = {"Mathematics": 1.0, "Physics": 1.2, "Chemistry": 1.0, "Biology": 0.6,
            "English": 0.2, "Computer Science": 0.8, "History": 0.3, "Economics": 0.7}
CORE = ["Mathematics", "English"]


def clip(x, lo, hi):
    return np.clip(x, lo, hi)


def make_students(rng, n):
    ability = rng.normal(0, 1, n)
    motivation = 0.3 * ability + rng.normal(0, 0.95, n)
    parent_edu = clip(np.round(2 + rng.normal(0, 1.1, n)), 0, 4)          # 0 none .. 4 postgraduate
    income = clip(np.round(0.6 * (parent_edu - 2) + 3 + rng.normal(0, 1, n)), 1, 5)
    grade_level = rng.choice([9, 10, 11, 12], n)
    return pd.DataFrame({
        "student_id": [f"S{i:04d}" for i in range(n)],
        "gender": rng.choice(["F", "M"], n),
        "grade_level": grade_level,
        "age": grade_level + 5 + rng.choice([0, 0, 0, 1], n),
        "parent_education": parent_edu.astype(int),
        "family_income_band": income.astype(int),
        "internet_access": (rng.random(n) < 0.55 + 0.08 * income).astype(int),
        "commute_minutes": clip(rng.gamma(2.2, 11, n), 3, 120).round().astype(int),
        "part_time_job": (rng.random(n) < clip(0.32 - 0.05 * income + 0.05 * (grade_level - 9), 0.02, 0.6)).astype(int),
        "extracurricular_hours": clip(rng.gamma(1.6, 1.6, n), 0, 14).round(1),
        "sleep_hours": clip(rng.normal(7.1, 1.0, n), 4, 10).round(1),
        "stress_level": clip(np.round(3 + 0.4 * (grade_level - 10) - 0.3 * motivation + rng.normal(0, 1, n)), 1, 5).astype(int),
        "health_status": clip(np.round(rng.normal(3.9, 0.9, n)), 1, 5).astype(int),
        "tutoring": (rng.random(n) < clip(0.05 + 0.06 * income, 0, 0.5)).astype(int),
        "past_failures": rng.choice([0, 1, 2, 3], n, p=[.78, .15, .05, .02]),
        "_ability": ability, "_motivation": motivation,
    })


def make_rows(rng, students):
    rows = []
    names = list(SUBJECTS)
    for _, s in students.iterrows():
        electives = rng.choice([x for x in names if x not in CORE], SUBJECTS_PER_STUDENT - len(CORE), replace=False)
        for subj in CORE + list(electives):
            aptitude = s._ability + rng.normal(0, 0.6)               # subject-specific talent
            interest = clip(np.round(3 + 0.5 * s._motivation + 0.4 * aptitude + rng.normal(0, 0.9)), 1, 5)
            teacher = rng.normal(0, 1)                               # teacher effectiveness (unobserved)
            class_size = int(clip(rng.normal(32, 6), 18, 48))
            study = clip(np.exp(0.9 + 0.35 * s._motivation + 0.12 * (interest - 3) - 0.25 * s.part_time_job
                                - 0.05 * s.extracurricular_hours + rng.normal(0, 0.45)), 0, 20)
            attendance = clip(92 + 4 * s._motivation - 0.08 * s.commute_minutes - 4 * s.part_time_job
                              + 1.5 * (s.health_status - 4) + rng.normal(0, 4), 40, 100)
            prev_year = clip(67 + 10 * aptitude - 3 * SUBJECTS[subj] + 1.5 * s.parent_education + rng.normal(0, 6), 0, 100)
            completion = clip(84 + 8 * s._motivation + 0.35 * (attendance - 90) - 3 * s.part_time_job + rng.normal(0, 7), 20, 100)
            late = clip(np.round((100 - completion) / 10 + rng.normal(0, 1.2)), 0, 10)
            assignment = clip(66 + 9 * aptitude + 4 * s._motivation + 0.12 * (completion - 85) + 2 * teacher
                              - 2 * SUBJECTS[subj] + rng.normal(0, 5), 0, 100)
            participation = clip(np.round(3 + 0.5 * s._motivation + 0.3 * (interest - 3) + rng.normal(0, 0.9)), 1, 5)
            quiz = clip(64 + 10 * aptitude + 6.6 * np.log1p(study) + 0.12 * (attendance - 90) - 3 * SUBJECTS[subj]
                        + rng.normal(0, 6), 0, 100)
            midterm = clip(62 + 11 * aptitude + 4 * np.log1p(study) + 0.15 * (attendance - 90) + 3 * s.tutoring
                           - 3.5 * SUBJECTS[subj] + 1.5 * teacher + rng.normal(0, 6.5), 0, 100)
            # sleeping less than ~7.8h costs marks, quadratically; more sleep gives no extra benefit
            sleep_penalty = -0.55 * (7.8 - s.sleep_hours) ** 2 if s.sleep_hours < 7.8 else 0.0
            final_exam = clip(60 + 11 * aptitude + 3.5 * s._motivation + 4.5 * np.log1p(study)
                              + 0.18 * (attendance - 90) + 2.5 * s.tutoring + sleep_penalty - 1.4 * (s.stress_level - 3)
                              - 4 * SUBJECTS[subj] + 2 * teacher + 0.075 * (midterm - 65)
                              - 2.5 * s.past_failures + rng.normal(0, 5), 0, 100)
            final_marks = clip(0.6 * final_exam + 0.2 * midterm + 0.2 * assignment, 0, 100)
            rows.append({
                "student_id": s.student_id, "subject": subj, "subject_interest": int(interest),
                "class_size": class_size, "prev_year_score": round(prev_year, 1),
                "study_hours_week": round(study, 1), "attendance_pct": round(attendance, 1),
                "assignment_completion_pct": round(completion, 1), "late_submissions": int(late),
                "assignment_avg_score": round(assignment, 1), "class_participation": int(participation),
                "quiz_avg_score": round(quiz, 1), "midterm_score": round(midterm, 1),
                "final_exam_score": round(final_exam, 1), "final_marks": round(final_marks, 1),
            })
    return pd.DataFrame(rows)


def generate(seed=SEED, n=N_STUDENTS):
    rng = np.random.default_rng(seed)
    students = make_students(rng, n)
    df = make_rows(rng, students).merge(students.drop(columns=["_ability", "_motivation"]), on="student_id")
    df["grade_letter"] = np.select([df.final_marks >= 85, df.final_marks >= 70, df.final_marks >= 55, df.final_marks >= 40],
                                   ["A", "B", "C", "D"], "F")
    df["passed"] = (df.final_marks >= 40).astype(int)
    df["data_source"] = "synthetic"
    first = ["student_id", "subject", "gender", "age", "grade_level"]
    return df[first + [c for c in df.columns if c not in first]]


if __name__ == "__main__":
    out = Path("data/raw/edupredict_synthetic.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    assert not df.isna().any().any() and df.final_marks.between(0, 100).all()
    df.to_csv(out, index=False)
    print(f"{len(df)} rows, {df.student_id.nunique()} students, {df.shape[1]} columns -> {out}")
