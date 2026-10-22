# EduPredict AI — Presentation outline (≈ 10 slides)

1. **Title** — EduPredict AI: predicting student marks with a NumPy-only ANN.
2. **Problem** — Can an ANN learn how student indicators relate to final marks and predict unseen students? Estimate, not guarantee.
3. **Data** — Synthetic (simulated) dataset: 2,000 students × 6 of 8 subjects = 12,000 rows, 32 columns: home, lifestyle, attendance, study hours, assignments, quizzes, mid-term, final marks. State clearly that it is not real student data.
4. **Preprocessing** — one-hot + standardise (train statistics only); 70/15/15 split grouped by student; target-derived columns excluded.
5. **Fundamentals lab** — live demo: McCulloch–Pitts (AND/OR), Hebb's rule, Delta rule on toy examples (weights and sums); then a Delta-rule neuron on the real student data: error falls from 16 to about 7.8 within a few hundred students, versus 6.26 for the 16-neuron network.
6. **ANN architecture** — diagram from `docs/architecture.md`: inputs → 16 ReLU → 1 linear; MSE.
7. **Learning** — forward pass, loss, backprop equations, gradient descent. Gradient check: error ≈ 1e-7 or better.
8. **Three models by timing** — early (25 inputs), during term (31), late (33). Results table: baseline RMSE 16.52; early 8.36 (R² 0.74); during term 6.26 (0.86); late 3.77 (0.95). Loss curve and actual-vs-predicted.
9. **Live dashboard demo** — switch Early / During term / Late; change attendance, study hours, mid-term; show prediction, grade, gauge, network internals, what-if curve (case D shows the models disagree).
10. **Limitations & conclusion** — synthetic data (model recovers what was put in); `late` contains target components; 2 neurons did as well as 16 and the best straight line matched the network (6.28 vs 6.26), so the data is near-linear; correlation not causation.

Demo script: open Predict → Early model, case A (prev. year 82, subject Maths) → 77.8; switch to *During term* and set attendance 96, study 8 h, completion 98 → 82.8; switch to *Late* and set mid-term 85, assignment 88 → 87.5; then try case D to show how term data changes the estimate; open "Inside the network".
