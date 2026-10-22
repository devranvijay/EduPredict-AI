# Architecture

## System overview

```
 synthetic_data.py ──► data/raw/edupredict_synthetic.csv
                              │
                       preprocessing.py ──► train.py ──► models/<variant>/{weights.npz, preprocessing.json}
                       encode, scale,       MLP.fit        results/{metrics.json, predictions.csv,
                       grouped split        + evaluate                training_history.csv}
                                                                  │
                                       app.py (Streamlit) ◄───────┘
                                       Predict · What-if · Results · Network internals · Fundamentals lab
```

## Network (EduPredict ANN)

```
 inputs (25 / 31 / 33)                   hidden layer (16)          output (1)

 prev_year_score ●─┐                       ┌──●  h1
 attendance_pct  ●─┤                       ├──●  h2
 study_hours     ●─┤   z1 = X·W1 + b1      │  ⋮    a1 = ReLU(z1)   ŷ = a1·W2 + b2
 quiz_avg_score  ●─┼────────────────────►  ├──●  h8  ────────────►  ● final marks (standardised)
 sleep, stress…  ●─┤   W1: (n_in × 16)     │  ⋮                     W2: (16 × 1)
 ⋮               ●─┤                       └──●  h16
 subject one-hot ●─┘
```

| Item | Value |
|---|---|
| Inputs | `early` 25, `during_term` 31, `late` 33: standardised numerics, 4 binary (gender, internet, part-time job, tutoring), 8-way one-hot `subject` |
| Hidden | 16 neurons, ReLU, He initialisation |
| Output | 1 neuron, linear; target `final_marks` (0–100), standardised with train mean/std |
| Loss | MSE |
| Optimiser | Mini-batch gradient descent, lr 0.01, batch 128, up to 300 epochs, early stopping (patience 25) on validation MSE |

## Mathematics

Forward: `z1 = X W1 + b1`, `a1 = max(0, z1)`, `ŷ = a1 W2 + b2`, `L = mean((ŷ − y)²)`

Backward (n = batch size):

```
dz2 = 2(ŷ − y)/n           dW2 = a1ᵀ dz2          db2 = Σ dz2
dz1 = (dz2 W2ᵀ) ⊙ 1[z1>0]  dW1 = Xᵀ dz1           db1 = Σ dz1
update: θ ← θ − lr · dθ
```

Verified against central finite differences (max relative error 2e-8 to 1.3e-7 across the three models; samples
whose pre-activation lies within the step of ReLU's kink at 0 are excluded, since ReLU is not differentiable there).

## Design decisions

- **Three variants by information timing.** Early, during-term and late inputs answer different questions ("who is at risk before the term?" vs "what will the final be given the mid-term?"). Inputs are added cumulatively.
- **Never-inputs.** `final_exam_score`, `grade_letter`, `passed` contain or derive from the target and are excluded; a test enforces this.
- **Student-grouped split.** Each student has 6 rows; splitting by `student_id` keeps all of them in one split.
- **No leakage in preprocessing.** Means, stds and category lists are fitted on the training split only and persisted in `preprocessing.json`; the dashboard reuses them.
- **Unseen categories raise an error** rather than silently encoding as zeros.
