# EduPredict synthetic dataset

**This data is simulated, not collected from real students.** It exists to give the ANN richer inputs
(attendance, assignments, quizzes, lifestyle) than the UCI data has. Any accuracy measured on it shows
that the pipeline can recover the relationships written into `src/synthetic_data.py`; it says nothing
about real students. Label it "synthetic" in every report and slide.

Regenerate (deterministic, seed 2026): `python -m src.synthetic_data` → `data/raw/edupredict_synthetic.csv`

**Size:** 12,000 rows = 2,000 students × 6 subjects (Mathematics and English for all, plus 4 of the other
6 electives: Physics, Chemistry, Biology, Computer Science, History, Economics). 32 columns, no missing values.

## Columns

| Group | Column | Meaning / range |
|---|---|---|
| Identity | `student_id`, `subject` | row key = student × subject |
| Student | `gender` (F/M), `age` (14–18), `grade_level` (9–12) | |
| Home | `parent_education` 0–4, `family_income_band` 1–5, `internet_access` 0/1, `commute_minutes` 3–120 | |
| Lifestyle | `part_time_job` 0/1, `extracurricular_hours` 0–14/wk, `sleep_hours` 4–10, `stress_level` 1–5, `health_status` 1–5, `tutoring` 0/1, `past_failures` 0–3 | |
| Subject context | `subject_interest` 1–5, `class_size` 18–48 | |
| **Early** (known at term start) | `prev_year_score` 0–100 | previous year's mark in the subject |
| **During term** | `study_hours_week`, `attendance_pct`, `assignment_completion_pct`, `late_submissions` 0–10, `assignment_avg_score`, `class_participation` 1–5, `quiz_avg_score` | |
| **Mid-term** | `midterm_score` 0–100 | |
| Target & parts | `final_marks` (0–100) = 0.6·`final_exam_score` + 0.2·`midterm_score` + 0.2·`assignment_avg_score` | |
| Derived from target | `final_exam_score`, `grade_letter` (A≥85, B≥70, C≥55, D≥40, else F), `passed` (≥40) | |
| Flag | `data_source` = `synthetic` | |

## Which columns to use as inputs

- **Never use** `final_exam_score`, `grade_letter`, `passed` as inputs: they contain or derive from the target (`final_exam_score` correlates 0.99 with `final_marks`).
- `midterm_score` and `assignment_avg_score` are components of `final_marks`, so including them is the synthetic equivalent of using G1/G2 in the UCI data. Train several variants, as before:
  1. **Early** — home, lifestyle, subject context, `prev_year_score` only
  2. **During term** — early + attendance, study hours, completion, quizzes, participation, late submissions
  3. **Late** — during term + `midterm_score`, `assignment_avg_score`
- Split by `student_id` (a student has 6 rows), not by row, or the same student leaks into train and test.

## How the data was generated (the "ground truth")

Each student has a hidden **ability** and **motivation** (correlated 0.3). Each student–subject pair adds
subject talent noise, a hidden teacher-effectiveness term and a subject difficulty (Physics 1.2 hardest,
English 0.2 easiest). Then:

- Attendance rises with motivation and health, falls with commute and part-time work.
- Study hours rise with motivation and interest, fall with job and extracurriculars (log-scale effect on marks).
- Assignment completion follows motivation and attendance; late submissions follow (100 − completion).
- Quiz, mid-term and final exam rise with ability, study hours, attendance and tutoring, and fall with stress, past failures, subject difficulty. Sleep below ~7.8 h costs marks quadratically.
- Noise (SD 5–6.5 marks) is added at each stage, so predictions can never be perfect.

**Columns with no direct effect on marks by construction:** `gender`, `internet_access`, `class_size`.
`age` and `grade_level` act only through stress and part-time work; income and parent education act only
through tutoring, part-time work and `prev_year_score`. Their raw correlation with `final_marks` is therefore
near zero (≈0.02), so do not present results as evidence about gender, income or internet access.

## Resulting distribution (seed 2026)

| Measure | Value |
|---|---|
| `final_marks` mean / SD / range | 62.0 / 15.9 / 6.4–100 |
| Pass rate (≥ 40) | 91.8% |
| Grades | A 8%, B 23%, C 36%, D 25%, F 8% |
| Mean marks by subject | English 63.9, History 63.6, Biology 62.7, Economics 62.1, CS 61.0, Maths 60.9, Physics 60.9, Chemistry 60.8 |
| Mean marks by stress level 1→5 | 69.1, 65.2, 62.2, 60.1, 57.2 |
| Mean marks by attendance <70 / 70–80 / 80–90 / 90+ | 36.7, 51.3, 58.8, 67.8 |

Correlations with `final_marks` were measured before the final baseline shift (mean raised from 54 to 62, which
does not change correlations materially): prev_year 0.81, quiz 0.86, midterm 0.90, assignment 0.92, study hours
0.43, attendance 0.39, completion 0.43, interest 0.56, late submissions −0.29, stress −0.20.

## Caveats

- Relationships are modelling assumptions (plausible, not measured); the model will "discover" exactly what was put in.
- Marks near 100 are clipped, and the distribution is smoother than real marks (no dropouts, no cheating, no exam-day effects).
