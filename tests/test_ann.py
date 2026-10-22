import numpy as np
import pytest

from src import learning_rules as lr
from src import preprocessing as pp
from src.ann import MLP, gradient_check
from src.evaluation import mae, rmse


@pytest.fixture(scope="module")
def data():
    df = pp.load_raw()
    idx = pp.group_split(df, seed=1)
    return df, idx


def test_gradient_check_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(30, 6)), rng.normal(size=30)
    assert gradient_check(MLP(6, 8, seed=1), X, y) < 1e-5


def test_network_learns_a_linear_target():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = X @ np.array([1.0, -2.0, 0.5])
    net = MLP(3, 16, seed=0)
    net.fit(X[:240], y[:240], X[240:], y[240:], lr=0.01, epochs=300)
    assert rmse(y[240:], net.predict(X[240:])) < 0.3 * y.std()


def test_save_load_roundtrip(tmp_path):
    net = MLP(4, 5, seed=3)
    net.save(tmp_path / "w.npz")
    X = np.random.default_rng(0).normal(size=(7, 4))
    assert np.allclose(net.predict(X), MLP.load(tmp_path / "w.npz").predict(X))


def test_split_is_disjoint_and_keeps_students_together(data):
    df, (tr, va, te) = data
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    assert len(tr) + len(va) + len(te) == len(df)
    ids = df["student_id"]
    assert not (set(ids.iloc[tr]) & set(ids.iloc[te]) or set(ids.iloc[tr]) & set(ids.iloc[va]))


def test_scaler_uses_train_statistics_only(data):
    df, (tr, _, te) = data
    cfg = pp.fit_config(df.iloc[tr])
    assert abs(pp.transform(df.iloc[tr], cfg)[:, 0].mean()) < 1e-9  # age column, train only
    assert cfg["target_mean"] == pytest.approx(df.iloc[tr][pp.TARGET].mean())


def test_variants_only_use_inputs_known_at_that_time(data):
    df, (tr, _, _) = data
    early = set(pp.column_names(pp.fit_config(df.iloc[tr], "early")))
    term = set(pp.column_names(pp.fit_config(df.iloc[tr], "during_term")))
    late = set(pp.column_names(pp.fit_config(df.iloc[tr], "late")))
    assert early < term < late
    assert not {"attendance_pct", "midterm_score"} & early
    assert "midterm_score" not in term and "midterm_score" in late


def test_target_and_derived_columns_are_never_inputs(data):
    df, (tr, _, _) = data
    names = " ".join(pp.column_names(pp.fit_config(df.iloc[tr], "late")))
    assert not any(c in names for c in pp.EXCLUDED)


def test_transform_rejects_unseen_category(data):
    df, (tr, _, _) = data
    cfg = pp.fit_config(df.iloc[tr])
    bad = df.iloc[tr].head(2).copy()
    bad["subject"] = "Astrology"
    with pytest.raises(ValueError):
        pp.transform(bad, cfg)


def test_metrics():
    assert mae([0, 10], [2, 6]) == 3.0
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))


def test_mp_neuron_implements_and_or():
    assert [r["output"] for r in lr.demo_mp("AND")] == [0, 0, 0, 1]
    assert [r["output"] for r in lr.demo_mp("OR")] == [0, 1, 1, 1]


def test_hebb_learns_bipolar_and():
    _, w, b = lr.demo_hebb()
    for x, t in [((1, 1), 1), ((1, -1), -1), ((-1, 1), -1), ((-1, -1), -1)]:
        assert np.sign(np.dot(w, x) + b) == t


def test_delta_rule_reduces_error_and_separates_and():
    _, losses, w, b = lr.demo_delta()
    assert losses[-1] < losses[0]
    assert [int(w @ np.array(x) + b >= 0.5) for x in [(0, 0), (0, 1), (1, 0), (1, 1)]] == [0, 0, 0, 1]


def test_delta_rule_on_data_recovers_linear_weights():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    true_w = np.array([1.5, -2.0, 0.5])
    y = X @ true_w + 0.3
    out = lr.delta_rule_fit(X[:300], y[:300], X[300:], y[300:], lr=0.02, epochs=15)
    assert not out["diverged"]
    assert out["val_mse"][-1] < 0.01 * out["val_mse"][0]
    assert np.allclose(out["weights"][-1], true_w, atol=0.1)
    cp = out["checkpoints"]
    assert cp[0]["updates"] == 0 and cp[-1]["updates"] == 15 * 300
    assert cp[-1]["val_mse"] < cp[0]["val_mse"]
    ls_w, ls_b = lr.least_squares(X, y)
    assert np.allclose(ls_w, true_w) and ls_b == pytest.approx(0.3)


def test_delta_rule_flags_divergence_with_huge_learning_rate():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(200, 5)), rng.normal(size=200)
    assert lr.delta_rule_fit(X, y, X, y, lr=5.0, epochs=40)["diverged"]
