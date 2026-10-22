"""EduPredict AI dashboard. Run: streamlit run app.py"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import learning_rules as lr
from src import preprocessing as pp
from src.ann import MLP
from src.evaluation import rmse

ROOT = Path(__file__).parent
VARIANTS = {"Early": "early", "During term": "during_term", "Late (+ mid-term)": "late"}
VARIANT_INFO = {
    "early": "Uses only background known before the term starts (home, lifestyle, last year's score). "
             "Most useful for early warning, least accurate.",
    "during_term": "Adds attendance, study hours, assignments and quizzes recorded during the term.",
    "late": "Adds the mid-term score and assignment average - most accurate, but available late. "
            "These are components of the final mark, so this model partly reads the answer.",
}
PROMINENT = ["subject", "prev_year_score", "attendance_pct", "study_hours_week", "assignment_completion_pct",
             "quiz_avg_score", "midterm_score", "assignment_avg_score", "sleep_hours", "stress_level"]
HELP = {
    "prev_year_score": "Last year's mark in this subject (0-100)",
    "attendance_pct": "Attendance in this subject (%)",
    "study_hours_week": "Self-study hours per week for this subject",
    "assignment_completion_pct": "Share of assignments submitted (%)",
    "quiz_avg_score": "Average quiz score (0-100)",
    "midterm_score": "Mid-term exam score (0-100) - a component of the final mark",
    "assignment_avg_score": "Average assignment score (0-100) - a component of the final mark",
    "sleep_hours": "Average nightly sleep",
    "stress_level": "Self-reported stress, 1 (low) to 5 (high)",
}

st.set_page_config(page_title="EduPredict AI", layout="wide")

GREEN, GREEN_LIGHT, INK = "#0071e3", "#e8f1fc", "#1d1d1f"  # Apple blue accent (names kept for gauge)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#d2d2d7",
                     "figure.facecolor": "none", "axes.facecolor": "none", "axes.prop_cycle":
                     plt.cycler(color=[GREEN, "#86868b", "#34c759"]), "font.size": 9})
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", Inter, "Segoe UI", system-ui, sans-serif; }
.stApp { background: radial-gradient(1200px 500px at 15% -10%, #e6efff 0%, transparent 60%),
                     radial-gradient(900px 500px at 100% 0%, #f1e8ff 0%, transparent 55%), #f5f5f7; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; max-width: 1200px; }
h1, h2, h3 { letter-spacing: -0.022em; line-height: 1.1; font-weight: 700; }
h3 { font-size: 1.35rem !important; letter-spacing: -0.01em; }
div[data-testid="stCaptionContainer"], .stCaption { letter-spacing: .005em; color: #6e6e73; }
/* translucent material cards */
div[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(255,255,255,.72); border: 1px solid rgba(255,255,255,.7);
    border-radius: 24px; padding: .8rem 1.1rem; box-shadow: 0 4px 24px rgba(0,0,0,.05);
    backdrop-filter: blur(20px) saturate(180%); -webkit-backdrop-filter: blur(20px) saturate(180%); }
.kpi { background: rgba(255,255,255,.72); border: 1px solid rgba(255,255,255,.7); border-radius: 24px; padding: 18px 22px;
    box-shadow: 0 4px 24px rgba(0,0,0,.05); backdrop-filter: blur(20px) saturate(180%); }
.kpi small { color:#6e6e73; font-size:.82rem; letter-spacing:.01em; }
.kpi h2 { margin:.15rem 0 0; font-size:2.1rem; letter-spacing:-0.03em; }
.kpi.hero { background: linear-gradient(145deg,#2997ff,#0071e3); color:#fff; border-color: transparent; }
.kpi.hero small { color: rgba(255,255,255,.85); }
.hello h1 { font-size: clamp(1.7rem, 2.8vw, 2.3rem); margin:0; line-height:1.1; letter-spacing:-0.03em; }
.hello p { margin:.5rem 0 0; font-size:1.15rem; color:#6e6e73; letter-spacing:-0.005em; }
/* segmented controls: even track padding, equal-width segments, white thumb */
div[data-testid="stButtonGroup"] { background: rgba(120,120,128,.12); border-radius: 999px; padding: 3px; width: fit-content; }
div[data-testid="stButtonGroup"] div:has(> button) { display: grid; grid-auto-flow: column; grid-auto-columns: 1fr; gap: 2px; }
div[data-testid="stButtonGroup"] button { border: none !important; border-radius: 999px !important; background: transparent !important;
    box-shadow: none !important; padding: 7px 22px !important; min-height: 0 !important; min-width: 9.5rem; flex: 1 1 0; justify-content: center; white-space: nowrap;
    transition: background 160ms ease-out, transform 100ms ease-out; }
div[data-testid="stButtonGroup"] button[aria-checked="true"], div[data-testid="stButtonGroup"] button[kind$="Active"] {
    background: #fff !important; box-shadow: 0 1px 4px rgba(0,0,0,.14) !important; font-weight: 600; }
.seglabel { font-size:.82rem; color:#6e6e73; letter-spacing:.01em; margin:0 0 6px 4px; }
button:active { transform: scale(.97); }
/* touch + keyboard polish: no 300ms tap delay or grey tap flash, no pull-to-refresh losing state, visible keyboard focus */
html, body { overscroll-behavior-y: contain; }
button, [role="button"], [data-baseweb="select"] { touch-action: manipulation; -webkit-tap-highlight-color: transparent; }
button:focus-visible { outline: 3px solid rgba(0,113,227,.45); outline-offset: 2px; }
@media (max-width: 640px) { input, textarea { font-size: 16px !important; } }  /* stops iOS zooming into fields */
@media (max-width: 768px) {  /* sidebar overlays the page here: near-opaque so the page gradient does not show through as a blob */
    section[data-testid="stSidebar"] { background: rgba(255,255,255,.94); box-shadow: 8px 0 32px rgba(0,0,0,.12); } }
/* spacing rhythm: 20px between blocks, roomy cards */
.block-container { padding-bottom: 4rem; }
div[data-testid="stVerticalBlock"], div[data-testid="stHorizontalBlock"] { gap: 1.25rem; }
div[data-testid="stVerticalBlockBorderWrapper"] { padding: 1.1rem 1.4rem; }
.kpi { padding: 18px 24px; }
.section h3 { margin: .25rem 0 0; font-size: 1.5rem !important; }
.section p { margin: .25rem 0 .25rem; color: #6e6e73; font-size: 1rem; letter-spacing: -0.005em; }
.note { color:#515154; font-size:.88rem; line-height:1.4; margin-top:.25rem; }
details[data-testid="stExpander"] { border: none; background: transparent; }
details[data-testid="stExpander"] summary { font-weight: 600; }
.hello small { color:#0071e3; font-weight:600; letter-spacing:.02em; font-size:.95rem; }
.hello h1 { margin-top:0; }
/* phones: tighter gutters, wrapping 2-column nav, smaller KPI type */
@media (max-width: 640px) {
    .block-container { padding: 3.5rem 1rem 3rem; }
    div[data-testid="stButtonGroup"] { width: 100%; border-radius: 20px; }
    div[data-testid="stButtonGroup"] div:has(> button) { grid-auto-flow: row; grid-template-columns: repeat(2, 1fr); }
    div[data-testid="stButtonGroup"] button { min-width: 0; padding: 8px 10px !important; white-space: normal; font-size: .85rem; }
    .kpi { padding: 16px 18px; border-radius: 20px; margin-bottom: .9rem; } .kpi h2 { font-size: 1.6rem; }
    .hello p { font-size: 1rem; }
    div[data-testid="stHorizontalBlock"]:has(.kpi) { flex-wrap: wrap; gap: .75rem !important; row-gap: 1rem !important; }
    div[data-testid="stHorizontalBlock"]:has(.kpi) > div[data-testid="stColumn"] { flex: 1 1 calc(50% - .75rem); min-width: calc(50% - .75rem); }
    div[data-testid="stElementContainer"]:has(div[data-testid="stButtonGroup"]), div[data-testid="stButtonGroup"] { width: 100% !important; }
}
/* sidebar: translucent material, vertical nav */
section[data-testid="stSidebar"] { background: rgba(255,255,255,.62); border-right: 1px solid rgba(0,0,0,.06);
    backdrop-filter: blur(24px) saturate(180%); -webkit-backdrop-filter: blur(24px) saturate(180%); }
/* sidebar is a full-height column: slim header row, brand + menu at the top, note pinned to the bottom */
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { display: flex; flex-direction: column; }
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] { flex: none; height: 2.75rem; min-height: 0; padding: .5rem 1rem 0; }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; padding: 0 1.25rem 1.25rem; }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div { flex: 1 1 auto; display: flex; flex-direction: column; }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] { flex: 1 1 auto; gap: .4rem; }
.brand { display: flex; align-items: center; gap: .75rem; padding: .25rem .25rem 1.1rem; margin-bottom: .5rem; border-bottom: 1px solid rgba(0,0,0,.07); }
.brand .logo { width: 38px; height: 38px; flex: none; border-radius: 11px; background: linear-gradient(145deg,#2997ff,#0071e3); color:#fff;
    display:flex; align-items:center; justify-content:center; font-weight:700; font-size:1.15rem; box-shadow: 0 4px 12px rgba(0,113,227,.3); }
.brand b { display:block; font-size: 1.15rem; letter-spacing: -0.02em; line-height:1.15; } .brand b span { color:#0071e3; }
.brand small { display:block; color:#6e6e73; font-size:.8rem; margin-top:.15rem; }
.menulabel { color:#86868b; font-size:.72rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; padding: .9rem .85rem .35rem; margin-bottom: .6rem; line-height: 1.2; }
section[data-testid="stSidebar"] button { justify-content: flex-start; text-align: left; border: none !important; box-shadow: none !important;
    border-radius: 12px !important; padding: .8rem 1rem !important; min-height: 2.9rem; background: transparent !important; color: #1d1d1f !important;
    font-weight: 500; transition: background 140ms ease-out, transform 100ms ease-out; }
section[data-testid="stSidebar"] button > div, section[data-testid="stSidebar"] button [data-testid="stMarkdownContainer"] {
    justify-content: flex-start !important; text-align: left !important; width: 100%; }
section[data-testid="stSidebar"] button p { text-align: left !important; }
section[data-testid="stSidebar"] button:hover { background: rgba(120,120,128,.12) !important; }
section[data-testid="stSidebar"] button[kind="primary"] { background: rgba(0,113,227,.12) !important; color: #0071e3 !important; font-weight: 600; }
div[data-testid="stElementContainer"]:has(.sidefoot) { margin-top: auto; margin-bottom: 1rem; }
.sidefoot { color:#6e6e73; font-size:.8rem; line-height:1.5; background: rgba(120,120,128,.09); border-radius: 14px; padding: .9rem 1rem; }
.sidefoot b { color:#1d1d1f; }
/* smoother navigation: no dimming of stale elements, no running-man flicker, gentle fade-in */
[data-stale="true"] { opacity: 1 !important; transition: none !important; }
div[data-testid="stStatusWidget"] { visibility: hidden; }
@keyframes fadein { from { opacity: 0; } to { opacity: 1; } }
.block-container > div[data-testid="stVerticalBlock"] > div { animation: fadein .22s ease-out; }
/* brand block is one big invisible button -> Overview */
.st-key-brand { position: relative; }
.st-key-brand div[data-testid="stElementContainer"]:has(button) { position: absolute; inset: 0; margin: 0; }
section[data-testid="stSidebar"] .st-key-brand button { width: 100%; height: 100%; min-height: 0 !important; padding: 0 !important; opacity: 0; cursor: pointer; }
.st-key-brand .logo { transition: transform 160ms ease-out; }
.st-key-brand:hover .logo { transform: scale(1.06); }
.st-key-brand:active .brand { opacity: .7; }
@media (prefers-reduced-motion: reduce) { .block-container > div[data-testid="stVerticalBlock"] > div { animation: none; } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
@media (prefers-reduced-transparency: reduce) {
    div[data-testid="stVerticalBlockBorderWrapper"], .kpi { background:#fff; backdrop-filter:none; } }
</style>""", unsafe_allow_html=True)


def kpi_card(col, label, value, hero=False):
    col.markdown(f'<div class="kpi{" hero" if hero else ""}"><small>{label}</small><h2>{value}</h2></div>',
                 unsafe_allow_html=True)


def gauge(marks):
    """Semicircle gauge of predicted marks (0-100)."""
    from matplotlib.patches import Wedge
    fig, ax = plt.subplots(figsize=(3.6, 2))
    ax.add_patch(Wedge((0, 0), 1, 0, 180, width=.28, color=GREEN_LIGHT))
    ax.add_patch(Wedge((0, 0), 1, 180 - 1.8 * marks, 180, width=.28, color=GREEN))
    ax.text(0, .12, f"{marks:.0f}%", ha="center", fontsize=20, fontweight="bold")
    ax.set(xlim=(-1.1, 1.1), ylim=(0, 1.1), aspect="equal")
    ax.axis("off")
    return fig


@st.cache_resource
def load_model(variant):
    folder = ROOT / "models" / variant
    return MLP.load(folder / "weights.npz"), pp.load_config(folder / "preprocessing.json")


def feature_widget(f, container, values):
    name = f["name"]
    if f["kind"] == "numeric":
        if f["integer"]:
            values[name] = container.slider(name, int(f["min"]), int(f["max"]), int(f["default"]),
                                            help=HELP.get(name), key=name)
        else:
            values[name] = container.slider(name, float(f["min"]), float(f["max"]), round(f["default"] * 2) / 2,
                                            step=0.5, help=HELP.get(name), key=name)
    else:
        opts = f["options"]
        values[name] = container.selectbox(name, opts, opts.index(f["default"]), help=HELP.get(name), key=name)


def predict_marks(net, cfg, values):
    return float(pp.unscale_target(net.predict(pp.transform(pd.DataFrame([values]), cfg)), cfg)[0])


def tab_predict():
    st.markdown('<div class="seglabel">Model</div>', unsafe_allow_html=True)
    choice = st.segmented_control("Model", list(VARIANTS), default=list(VARIANTS)[0], key="model",
                                  label_visibility="collapsed") or list(VARIANTS)[0]
    net, cfg = load_model(VARIANTS[choice])
    st.caption(VARIANT_INFO[cfg["variant"]])
    kpi = st.columns(4)
    left, right = st.columns([3, 2])
    values = {}
    feats = {f["name"]: f for f in cfg["features"]}
    shown = [n for n in PROMINENT if n in feats]
    with left.container(border=True):
        st.subheader("Student details")
        cols = st.columns(2)
        for i, n in enumerate(shown):
            feature_widget(feats[n], cols[i % 2], values)
        with st.expander("More student attributes (defaults = most typical training value)"):
            cols = st.columns(3)
            for i, n in enumerate(n for n in feats if n not in shown):
                feature_widget(feats[n], cols[i % 3], values)
    marks = float(np.clip(predict_marks(net, cfg, values), 0, 100))
    t = load_results()[0][VARIANTS[choice]]
    kpi_card(kpi[0], "Predicted final marks", f"{marks:.1f} / 100", hero=True)
    kpi_card(kpi[1], "Predicted grade", pp.grade_letter(marks))
    kpi_card(kpi[2], "Model test MAE (marks)", f"±{t['test']['ann']['mae']:.1f}")
    kpi_card(kpi[3], "Model test R²", f"{t['test']['ann']['r2']:.2f}")
    with right.container(border=True):
        st.subheader("Predicted final marks")
        st.pyplot(gauge(marks))
        st.caption(f"Typical error of this model: see Results tab. "
                   "A statistical estimate from past patterns, not a guarantee.")
        st.subheader("What-if explorer")
        vary = st.selectbox("Vary one input", [n for n in shown if feats[n]["kind"] == "numeric"])
        grid = np.arange(int(feats[vary]["min"]), int(feats[vary]["max"]) + 1)
        curve = np.clip(pp.unscale_target(net.predict(pp.transform(pd.DataFrame([{**values, vary: v} for v in grid]), cfg)), cfg), 0, 100)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(grid, curve, marker="o")
        ax.axvline(values[vary], color="grey", ls=":")
        ax.set(xlabel=vary, ylabel="predicted marks", ylim=(0, 100))
        st.pyplot(fig)
        st.caption("Shows how the MODEL's output changes. It does not prove that changing this input "
                   "would cause a real student's marks to change.")
    network_panel(net, cfg, values)


def network_panel(net, cfg, values):
    with st.expander("Inside the network for this student"):
        x = pp.transform(pd.DataFrame([values]), cfg)
        y_hat, (_, z1, a1) = net.forward(x)
        st.markdown(f"**Input** ({x.shape[1]} scaled features) → **hidden** `z1 = x·W1 + b1` (16 values) → "
                    "**ReLU** `a1 = max(0, z1)` → **output** `ŷ = a1·W2 + b2` (linear).")
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.bar(range(16), a1.ravel())
        ax.set(xlabel="hidden neuron", ylabel="activation a1", title=f"{int((a1 > 0).sum())} of 16 neurons active")
        st.pyplot(fig)
        total = float((a1 @ net.params["W2"])[0, 0])
        b2 = float(net.params["b2"][0, 0])
        st.write(f"ŷ (standardised) = Σ a1·W2 + b2 = {total:.3f} + ({b2:.3f}) = {float(y_hat[0, 0]):.3f} → "
                 f"× {cfg['target_std']:.2f} + {cfg['target_mean']:.2f} = "
                 f"**{float(pp.unscale_target(y_hat, cfg)[0, 0]):.1f} marks**")


def tab_results():
    metrics, preds, hist = load_results()
    st.subheader("Test-set metrics (marks on 0-100; test set used once, at the end)")
    rows = []
    for label, v in VARIANTS.items():
        t = metrics[v]["test"]
        rows.append({"Model": label, "MAE": t["ann"]["mae"], "RMSE": t["ann"]["rmse"], "R²": t["ann"]["r2"],
                     "Baseline MAE (predict mean)": t["baseline_mean"]["mae"],
                     "Baseline RMSE": t["baseline_mean"]["rmse"]})
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True)
    st.markdown('<div class="seglabel">Show results for</div>', unsafe_allow_html=True)
    v = VARIANTS[st.segmented_control("Show results for", list(VARIANTS), default=list(VARIANTS)[0], key="res",
                                      label_visibility="collapsed") or list(VARIANTS)[0]]
    c1, c2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    h = hist[hist.variant == v]
    ax.plot(h.epoch, h.train_loss, label="train")
    ax.plot(h.epoch, h.val_loss, label="validation")
    ax.set(xlabel="epoch", ylabel="MSE (standardised marks)", title="Training loss")
    ax.legend()
    c1.pyplot(fig)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(preds["actual"], preds[f"pred_{v}"], s=14, alpha=.6)
    ax.plot([0, 100], [0, 100], "k:")
    ax.set(xlabel="actual marks", ylabel="predicted marks", title="Test set: actual vs predicted")
    c2.pyplot(fig)
    table = preds[["student_id", "subject", "prev_year_score", "attendance_pct", "study_hours_week", "actual", f"pred_{v}"]].copy()
    table["error"] = table[f"pred_{v}"] - table["actual"]
    st.dataframe(table.round(1), hide_index=True)
    model_insights(v, [k for k, x in VARIANTS.items() if x == v][0])
    st.caption(f"Backprop vs numerical gradient, max relative error = "
               f"{metrics[v]['gradient_check_max_rel_error']:.1e}; trained {metrics[v]['epochs_run']} epochs "
               "(early stopping on validation loss).")


@st.cache_data
def _load_results(mtime):
    r = ROOT / "results"
    return (json.loads((r / "metrics.json").read_text()), pd.read_csv(r / "predictions.csv"),
            pd.read_csv(r / "training_history.csv"))


def load_results():
    return _load_results((ROOT / "results" / "metrics.json").stat().st_mtime)


@st.cache_data
def load_dataset():
    return pd.read_csv(ROOT / pp.DATA_PATH)


@st.cache_data
def permutation_importance(variant):
    """RMSE increase on the test split when one input is shuffled (bigger = model relies on it more)."""
    net, cfg = load_model(variant)
    df = pd.read_csv(ROOT / "data" / "processed" / "dataset_with_split.csv")
    test = df[df.split == "test"].reset_index(drop=True)
    y, rng = test[pp.TARGET].to_numpy(), np.random.default_rng(0)
    base = rmse(y, pp.unscale_target(net.predict(pp.transform(test, cfg)), cfg))
    out = {}
    for f in cfg["features"]:
        shuffled = test.assign(**{f["name"]: rng.permutation(test[f["name"]].to_numpy())})
        out[f["name"]] = rmse(y, pp.unscale_target(net.predict(pp.transform(shuffled, cfg)), cfg)) - base
    return pd.Series(out).sort_values(ascending=False)


def section(title, sub=""):
    st.markdown(f'<div class="section"><h3>{title}</h3><p>{sub}</p></div>', unsafe_allow_html=True)


def chart_card(col, title, draw, height=2.8, note=None):
    """White card: title, matplotlib chart drawn by draw(ax), optional one-line takeaway."""
    with col.container(border=True):
        st.markdown(f"**{title}**")
        fig, ax = plt.subplots(figsize=(4.2, height))
        draw(ax)
        fig.tight_layout()
        st.pyplot(fig)
        if note:
            st.markdown(f'<div class="note">{note}</div>', unsafe_allow_html=True)


def binned_mean(df, col, bins, labels, target="final_marks"):
    return df.groupby(pd.cut(df[col], bins, labels=labels), observed=True)[target].mean()


def label(i):
    return f"{i.left:g}-{i.right:g}" if isinstance(i, pd.Interval) else str(i)


def bar(series, ax, rot=0, fmt="{:.0f}"):
    bars = ax.bar([label(i) for i in series.index], series.to_numpy(), color=GREEN, width=.62)
    ax.bar_label(bars, labels=[fmt.format(v) for v in series], fontsize=8, padding=3)
    ax.tick_params(axis="x", rotation=rot, labelsize=8)
    ax.set_yticks([])
    ax.spines[["left"]].set_visible(False)


def rise_note(m):
    return f"Average marks go from <b>{m.iloc[0]:.0f}</b> ({m.index[0]}) to <b>{m.iloc[-1]:.0f}</b> ({m.index[-1]})."


def hbar(series, ax):
    bars = ax.barh(series.index, series.to_numpy(), color=GREEN, height=.6)
    ax.bar_label(bars, labels=[f"{v:.0f}" for v in series], fontsize=8, padding=3)
    ax.set_xlim(0, 100)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["bottom"]].set_visible(False)


def more_insights(df):
    r1 = st.columns(3)
    chart_card(r1[0], "Grade distribution (% of rows)",
               lambda ax: bar(df.grade_letter.value_counts(normalize=True).reindex(list("ABCDF")).fillna(0) * 100, ax))
    chart_card(r1[1], "Pass rate by grade level (%)",
               lambda ax: (bar(df.groupby("grade_level").passed.mean() * 100, ax), ax.set_ylim(0, 112)))
    chart_card(r1[2], "Stress level vs average marks",
               lambda ax: (bar(df.groupby("stress_level").final_marks.mean(), ax), ax.set_xlabel("stress (1 low - 5 high)")))
    r2 = st.columns([1, 2])

    def sleep(ax):
        m = binned_mean(df, "sleep_hours", [3, 6, 7, 8, 9, 11], ["<6 h", "6-7 h", "7-8 h", "8-9 h", "9+ h"])
        ax.plot([label(i) for i in m.index], m.to_numpy(), marker="o", color=GREEN)
        ax.set(xlabel="sleep hours", ylim=(0, 100))
        ax.tick_params(axis="x", labelsize=8)

    def heatmap(ax):
        cols = ["final_marks", "prev_year_score", "midterm_score", "assignment_avg_score", "quiz_avg_score",
                "attendance_pct", "study_hours_week", "assignment_completion_pct", "late_submissions",
                "subject_interest", "sleep_hours", "stress_level"]
        c = df[cols].corr()
        ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(cols)), cols, fontsize=7)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{c.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6)
        ax.spines[:].set_visible(False)

    chart_card(r2[0], "Sleep vs average marks", sleep, height=3.8)
    chart_card(r2[1], "How the columns relate (1 = move together, -1 = opposite)", heatmap, height=3.8)


def tab_overview():
    data = load_dataset()
    f1, f2, f3 = st.columns([2, 2, 1.4], vertical_alignment="bottom")
    subjects = f1.multiselect("Subjects", sorted(data.subject.unique()), placeholder="All subjects") or sorted(data.subject.unique())
    grades = f2.multiselect("Grade level", sorted(data.grade_level.unique()), placeholder="All grade levels") or sorted(data.grade_level.unique())
    df = data[data.subject.isin(subjects) & data.grade_level.isin(grades)]
    f3.caption(f"{len(df):,} of {len(data):,} rows · synthetic data")
    if df.empty:
        st.info("No rows match these filters.")
        return
    k = st.columns(4)
    kpi_card(k[0], "Students", f"{df.student_id.nunique():,}", hero=True)
    kpi_card(k[1], "Average final marks", f"{df.final_marks.mean():.1f}")
    kpi_card(k[2], "Pass rate (marks 40+)", f"{df.passed.mean() * 100:.1f}%")
    kpi_card(k[3], "At risk (marks under 50)", f"{(df.final_marks < 50).mean() * 100:.1f}%")

    section("What moves marks", "The three factors with the clearest link to final marks.")
    subj = df.groupby("subject").final_marks.mean().sort_values()
    att = binned_mean(df, "attendance_pct", [0, 70, 80, 90, 101], ["<70%", "70-80%", "80-90%", "90%+"])
    study = binned_mean(df, "study_hours_week", [-1, 2, 4, 6, 9, 21], ["<2 h", "2-4 h", "4-6 h", "6-9 h", "9+ h"])
    c = st.columns(3)
    chart_card(c[0], "Average marks by subject",
               lambda ax: hbar(subj, ax),
               note=f"<b>{subj.index[-1]}</b> is highest ({subj.iloc[-1]:.0f}), <b>{subj.index[0]}</b> lowest ({subj.iloc[0]:.0f}).")
    chart_card(c[1], "Attendance vs average marks",
               lambda ax: bar(att, ax), note=rise_note(att))
    chart_card(c[2], "Study hours per week vs average marks",
               lambda ax: bar(study, ax), note=rise_note(study))

    section("Go deeper", "Optional extra views.")
    if st.toggle("Show more insights: grades, stress, sleep and correlations"):
        more_insights(df)


def model_insights(v, variant_label):
    imp = permutation_importance(v).head(10)[::-1]
    preds = load_results()[1]
    resid = preds[f"pred_{v}"] - preds["actual"]
    c1, c2 = st.columns(2)

    def draw_imp(ax):
        ax.barh(imp.index, imp.to_numpy(), color=GREEN)
        ax.set_xlabel("RMSE increase when shuffled")
        ax.tick_params(labelsize=7)

    def draw_res(ax):
        ax.hist(resid, bins=40, color=GREEN)
        ax.axvline(0, color="grey", ls=":")
        ax.set(xlabel="prediction error (marks)", ylabel="rows")

    chart_card(c1, f"What the {variant_label} model relies on", draw_imp, height=3.4)
    chart_card(c2, f"Prediction errors (mean {resid.mean():+.2f}, SD {resid.std():.2f})", draw_res, height=3.4)
    st.caption("Permutation importance on the test rows: shuffle one input and see how much the error grows. "
               "Correlated inputs share credit, and on synthetic data this mirrors the generator.")


@st.cache_data
def delta_on_data(features, lr_value, epochs):
    """Train one linear neuron with the Delta rule on the training split; evaluate on val/test (marks units)."""
    df = pd.read_csv(ROOT / "data" / "processed" / "dataset_with_split.csv")
    cfg = pp.load_config(ROOT / "models" / "during_term" / "preprocessing.json")
    stats = {f["name"]: f for f in cfg["features"]}
    parts = {k: df[df.split == k] for k in ("train", "val", "test")}
    X = {k: np.column_stack([(d[n] - stats[n]["mean"]) / stats[n]["std"] for n in features]) for k, d in parts.items()}
    y = {k: pp.scale_target(d[pp.TARGET], cfg) for k, d in parts.items()}
    fit = lr.delta_rule_fit(X["train"], y["train"], X["val"], y["val"], lr=lr_value, epochs=epochs)
    ls_w, ls_b = lr.least_squares(X["train"], y["train"])
    std = cfg["target_std"]
    rmse_marks = lambda w, b, k: float(np.sqrt(np.mean((X[k] @ w + b - y[k]) ** 2)) * std)
    return {"fit": fit, "std": std, "ls_w": ls_w, "ls_b": ls_b,
            "updates": [c["updates"] for c in fit["checkpoints"]],
            "val_rmse_curve": [float(np.sqrt(c["val_mse"]) * std) for c in fit["checkpoints"]],
            "weight_curve": np.array([c["w"] for c in fit["checkpoints"]]) * std,
            "delta_test": None if fit["diverged"] else rmse_marks(fit["weights"][-1], fit["biases"][-1], "test"),
            "ls_val": rmse_marks(ls_w, ls_b, "val"), "ls_test": rmse_marks(ls_w, ls_b, "test")}


def delta_on_student_data():
    st.markdown("A single neuron learns to predict final marks from the factors you pick, using the **Delta rule** "
                "(`w ← w + η·(target − output)·x`, one student at a time). Compare it with the best possible "
                "straight-line fit and with the 16-neuron network.")
    metrics = load_results()[0]["during_term"]
    options = [f["name"] for f in pp.load_config(ROOT / "models" / "during_term" / "preprocessing.json")["features"]
               if f["kind"] == "numeric"]
    c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="bottom")
    feats = c1.multiselect("Inputs for the neuron", options, default=["attendance_pct", "quiz_avg_score", "study_hours_week"])
    eta = c2.slider("Learning rate η", 0.001, 0.03, 0.01, 0.001, format="%.3f")
    epochs = c3.slider("Epochs", 1, 30, 10)
    if not feats:
        st.info("Pick at least one input.")
        return
    r = delta_on_data(tuple(feats), eta, epochs)
    fit = r["fit"]
    if fit["diverged"]:
        st.warning("The weights blew up: the learning rate is too high for this many inputs. Lower η.")
        return
    k = st.columns(4)
    kpi_card(k[0], "Delta neuron (test RMSE)", f"{r['delta_test']:.2f}", hero=True)
    kpi_card(k[1], "Best straight line", f"{r['ls_test']:.2f}")
    kpi_card(k[2], "16-neuron network", f"{metrics['test']['ann']['rmse']:.2f}")
    kpi_card(k[3], "Predict the mean", f"{metrics['test']['baseline_mean']['rmse']:.2f}")
    st.caption("Average error in marks on unseen test rows (lower is better). The network uses all 31 during-term inputs; "
               "the neuron only the ones picked, so this is not a like-for-like contest.")

    left, right = st.columns(2)

    def curve(ax):
        ax.plot(r["updates"], r["val_rmse_curve"], marker="o", ms=3, color=GREEN, label="Delta neuron")
        ax.axhline(r["ls_val"], color="#34c759", ls="--", label="best straight line")
        ax.axhline(metrics["val"]["ann"]["rmse"], color="#86868b", ls=":", label="16-neuron network")
        ax.set(xscale="symlog", xlabel="students seen (updates)", ylabel="validation error (marks)")
        ax.set_xlim(left=0)
        ax.legend(fontsize=7)

    def weights(ax):
        for j, n in enumerate(feats):
            ax.plot(r["updates"], r["weight_curve"][:, j], label=n)
        ax.set(xscale="symlog", xlabel="students seen (updates)", ylabel="marks per +1 SD of input")
        ax.set_xlim(left=0)
        ax.legend(fontsize=7)

    chart_card(left, "The error falls fast, then flattens", curve, height=3.2,
               note="After a few hundred students it is close to the best straight line, the most one neuron can do. "
                    "A large learning rate makes it bounce around instead of settling.")
    chart_card(right, "The weights find their level", weights, height=3.2,
               note="Each line: how many marks the neuron adds for a student one standard deviation above average on that input.")

    table = pd.DataFrame({"Input": feats,
                          "Learned (marks per +1 SD)": np.array(fit["weights"][-1]) * r["std"],
                          "Best straight line": r["ls_w"] * r["std"]}).sort_values("Learned (marks per +1 SD)", ascending=False)
    st.dataframe(table.round(2), hide_index=True)
    with st.expander("See the first 5 updates, step by step"):
        rows = []
        for i, step in enumerate(fit["steps"], 1):
            rows.append({"update": i, "output": round(step["output"], 3), "target": round(step["target"], 3),
                         "error": round(step["error"], 3),
                         **{f"w[{n}]": round(float(w), 4) for n, w in zip(feats, step["w"])}})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("Values are standardised. Each row: output = w·x + b, error = target − output, then every weight moves by η·error·input.")
    st.caption("A neuron can only draw a straight-line rule. The 16-neuron network adds a hidden layer and uses "
               "backpropagation, which spreads the same error signal through the layers with the chain rule. "
               "If the straight line comes close to the network, the data is close to linear.")


def tab_fundamentals():
    data_tab, toy_tab = st.tabs(["On the student data", "Toy examples"])
    with data_tab:
        delta_on_student_data()
    with toy_tab:
        toy_examples()


def toy_examples():
    st.subheader("McCulloch-Pitts neuron")
    st.caption("Binary inputs, fixed weights, fires when weighted sum ≥ threshold.")
    st.dataframe(pd.DataFrame(lr.demo_mp(st.radio("Gate", ["AND", "OR"], horizontal=True))), hide_index=True)
    st.subheader("Hebb's rule")
    st.caption("Δw = x·y, Δb = y: weights grow when input and output are active together (bipolar AND).")
    rows, w, b = lr.demo_hebb()
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.write(f"Final weights {w.astype(int).tolist()}, bias {int(b)}")
    st.subheader("Delta rule")
    st.caption("w ← w + η(t − y)x on a linear neuron; the error shrinks as weights are adjusted (learning AND).")
    eta = st.slider("Learning rate η", 0.01, 0.5, 0.1, 0.01)
    steps, losses, _, _ = lr.demo_delta(lr=eta)
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.plot(range(1, len(losses) + 1), losses)
    ax.set(xlabel="epoch", ylabel="MSE")
    st.pyplot(fig)
    st.dataframe(pd.DataFrame(steps), hide_index=True)
    st.info("Backpropagation (used by EduPredict) extends this error-driven idea through hidden layers via "
            "the chain rule; it is not identical to the single-neuron Delta rule.")


PAGES = {  # name -> (render function, icon, one-line description)
    "Overview": (tab_overview, ":material/space_dashboard:", ""),
    "Predict": (tab_predict, ":material/query_stats:", "Describe a student and get an estimate of the final marks."),
    "Results": (tab_results, ":material/analytics:", "How well each model does on students it has never seen."),
    "ANN fundamentals lab": (tab_fundamentals, ":material/school:",
                             "The classic neuron models behind the network, including a Delta-rule neuron trained on the student data."),
}


def go(name):
    st.session_state["page"] = name


# One click listener for the whole session (guarded so reruns do not stack it): after any sidebar nav click it
# collapses the sidebar on phone/tablet widths (Streamlit overlays it below 768px) and scrolls back to the top.
NAV_SCRIPT = """<script>
(() => {
  if (window.__eduNav) return;
  window.__eduNav = true;
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('section[data-testid="stSidebar"] button');
    if (!btn || btn.closest('[data-testid="stSidebarCollapseButton"]')) return;
    setTimeout(() => {
      if (window.matchMedia('(max-width: 768px)').matches) {
        const c = document.querySelector('[data-testid="stSidebarCollapseButton"] button');
        if (c) c.click();
      }
      const m = document.querySelector('[data-testid="stMain"]');
      if (m) m.scrollTo({top: 0, behavior: 'smooth'});
    }, 120);
  }, true);
})();
</script>"""


def sidebar():
    st.session_state.setdefault("page", "Overview")
    with st.sidebar:
        st.html(NAV_SCRIPT, unsafe_allow_javascript=True)  # first thing rendered, so it is live before the page body
        with st.container(key="brand"):
            st.markdown('<div class="brand"><div class="logo">E</div><div><b>Edu<span>Predict</span> AI</b>'
                        '<small>Student marks prediction</small></div></div>', unsafe_allow_html=True)
            st.button("Go to Overview", key="brand_home", on_click=go, args=("Overview",))
        st.markdown('<div class="menulabel">Menu</div>', unsafe_allow_html=True)
        for name, (_, icon, _) in PAGES.items():
            st.button(name, key=f"nav_{name}", icon=icon, use_container_width=True,
                      type="primary" if st.session_state["page"] == name else "secondary",
                      on_click=go, args=(name,))
        st.markdown('<div class="sidefoot"><b>Synthetic data</b><br>2,000 simulated students, 8 subjects.<br>'
                    'Estimates, not guarantees.</div>', unsafe_allow_html=True)


def main():
    sidebar()
    page = st.session_state["page"]
    render, _, subtitle = PAGES[page]
    if page == "Overview":
        st.markdown('<div class="hello"><h1>Predict final marks. See what moves them.</h1></div>',
                    unsafe_allow_html=True)
    else:
        section(page, subtitle)
    render()

main()
