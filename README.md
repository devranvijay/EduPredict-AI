# EduPredict AI

**Student Academic Performance Prediction Using a NumPy-Based Artificial Neural Network**

A one-hidden-layer ANN (16 ReLU neurons, linear output) written from scratch in NumPy — forward pass,
MSE loss, backpropagation, gradient descent — trained on a **synthetic** student dataset (8 subjects,
2,000 simulated students, 12,000 student–subject rows, 32 columns) and served through a Streamlit dashboard.
Scikit-learn is used only for the train/val/test split.

> The data is simulated (`src/synthetic_data.py`, see [docs/synthetic_dataset.md](docs/synthetic_dataset.md)),
> not real student records. Results show the pipeline recovers the relationships written into the generator.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
python -m src.synthetic_data     # (re)generate data/raw/edupredict_synthetic.csv (deterministic)
python train.py                  # trains three models, writes models/ and results/
streamlit run app.py             # dashboard
python -m pytest                 # tests (incl. numerical gradient check)
python -m src.learning_rules     # McCulloch-Pitts / Hebb / Delta rule printouts
```

## Layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard: predict, what-if, results, network internals, fundamentals lab |
| `train.py` | Student-grouped split, train, evaluate, save models and results |
| `src/synthetic_data.py` | Generator for the synthetic dataset |
| `src/ann.py` | NumPy MLP, backprop, gradient descent, gradient check |
| `src/preprocessing.py` | Loading, encoding, scaling (train-only stats), variants, grouped split |
| `src/evaluation.py` | MAE, RMSE, R² |
| `src/learning_rules.py` | McCulloch-Pitts, Hebb, Delta rule demos, plus a Delta-rule neuron trained on the student data |
| `models/<variant>/` | `weights.npz`, `preprocessing.json` for `early`, `during_term`, `late` |
| `results/` | `metrics.json`, `predictions.csv`, `training_history.csv`, `training_overview.png` |
| `docs/` | [architecture](docs/architecture.md), [report](docs/REPORT.md), [dataset](docs/synthetic_dataset.md), [presentation](docs/PRESENTATION.md), [viva](docs/VIVA.md) |

## Three models: when is the information available?

| Variant | Inputs (count) | Adds |
|---|---|---|
| `early` | 25 | home, lifestyle, subject, last year's score |
| `during_term` | 31 | + attendance, study hours, assignments, quizzes, participation |
| `late` | 33 | + mid-term and assignment averages (components of the final mark) |

## Headline results (test set: 1,800 rows / 300 unseen students, marks 0–100)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Predict-the-mean baseline | 13.13 | 16.52 | 0.00 |
| ANN `early` | 6.64 | 8.36 | 0.74 |
| ANN `during_term` | 4.99 | 6.26 | 0.86 |
| ANN `late` | 3.01 | 3.77 | 0.95 |

Predictions are statistical estimates, not guarantees, and the what-if explorer is not causal.
The first version of this project used the UCI *Student Performance* data; it was replaced by the synthetic
set because UCI has no attendance or assignment columns.
