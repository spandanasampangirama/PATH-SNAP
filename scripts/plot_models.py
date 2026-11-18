"""Model-comparison plots for report §3.3, drawn ONLY from committed result CSVs.  Owner: P2

Inputs (written by scripts/train_models.py and scripts/bayes_opt.py):
    results/model_comparison.csv, results/cv_folds.csv, results/training_history/*.csv,
    results/tuned_vs_untuned.csv (optional), results/optuna/trials.csv (optional)
Outputs (results/plots/):
    p2_cv_macro_f1.png            CV macro-F1 per model: mean bar + each fold as a dot (stability)
    p2_test_top1_macro_f1.png     student-test top-1 accuracy and macro-F1 side by side
    p2_test_log_loss.png          student-test log loss (lower is better)
    p2_inference_time.png         inference ms per 1k GPS points
    p2_confusion_<model>.png      student-test confusion matrix per model
    p2_model_<model>.png          per-model 4 panels: CV per fold, calibration, top-1 per test trip,
                                  permutation importance (needs scripts/model_diagnostics.py first)
    p2_mlp_loss_curve.png         MLP epoch history (the only model with one)
    p2_tuned_vs_untuned.png       (if tuned_vs_untuned.csv exists)
    p2_optuna_history.png         (if results/optuna/trials.csv exists)

Run:  python -m scripts.plot_models
Every plot is regenerated from the CSVs, so the report figures always match the recorded numbers.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pathsnap.config import CFG  # noqa: E402

# Colours: grey for data, one blue accent for the model the chart's point is about; orange only as the
# second series where two measures sit side by side.
GREY, BLUE, ORANGE = "#9a9994", "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.dpi": 130, "savefig.bbox": "tight",
})


def _title(ax, title: str, subtitle: str):
    ax.set_title(f"{title}\n", loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


def _bar_colors(models: pd.Series, accent: str) -> list[str]:
    return [BLUE if m == accent else GREY for m in models]


def _value_labels(ax, bars, fmt="{:.3f}"):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(b.get_height()),
                ha="center", va="bottom", fontsize=8.5, color=INK)


def plot_cv(cmp: pd.DataFrame, folds: pd.DataFrame, out: Path):
    """Dot plot (not bars): the models differ by a few points, so the axis is zoomed, and a zoomed
    axis is only honest with position marks. Each fold is a small dot, the mean a bold tick + label."""
    d = cmp.sort_values("cv_macro_f1_mean", ascending=True)
    best = d.iloc[-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    for i, r in enumerate(d.itertuples()):
        c = BLUE if r.model == best.model else GREY
        f = folds.loc[folds.model == r.model, "macro_f1"]
        ax.scatter(f, np.full(len(f), i), s=18, color=c, alpha=0.55, zorder=2,
                   label="single fold" if i == 0 else None)
        ax.scatter([r.cv_macro_f1_mean], [i], s=160, marker="|", color=c, linewidths=2.5, zorder=3,
                   label="mean of folds" if i == 0 else None)
        ax.text(f.max() + 0.003, i, f"{r.cv_macro_f1_mean:.3f} ± {r.cv_macro_f1_std:.3f}", va="center",
                fontsize=8.5, color=INK)
    ax.set_yticks(range(len(d)), d["model"])
    ax.set_xlim(folds.macro_f1.min() - 0.01, folds.macro_f1.max() + 0.03)
    ax.set_xlabel("macro-F1 on held-out trips")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    _title(ax, f"{best.model} has the highest mean CV macro-F1 ({best.cv_macro_f1_mean:.3f})",
           f"GroupKFold by trip, {CFG['cv']['n_splits']} folds, train partition. Dots = individual folds.")
    fig.savefig(out / "p2_cv_macro_f1.png"); plt.close(fig)


def plot_test(cmp: pd.DataFrame, out: Path):
    """Dot plot of the two student-test scores per model (zoomed axis -> position marks, not bars)."""
    d = cmp.sort_values("test_top1_accuracy", ascending=True)
    best = d.iloc[-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    y = np.arange(len(d))
    ax.hlines(y, d.test_top1_accuracy, d.test_macro_f1, color=GRID, lw=2, zorder=1)
    ax.scatter(d.test_top1_accuracy, y, s=55, color=BLUE, zorder=3, label="top-1 accuracy (per GPS point)")
    ax.scatter(d.test_macro_f1, y, s=55, color=ORANGE, marker="D", zorder=3, label="macro-F1 (per candidate)")
    for yi, r in zip(y, d.itertuples()):
        lo_v, hi_v = sorted([r.test_top1_accuracy, r.test_macro_f1])
        ax.text(lo_v - 0.004, yi, f"{lo_v:.3f}", ha="right", va="center", fontsize=8.5, color=INK)
        ax.text(hi_v + 0.004, yi, f"{hi_v:.3f}", ha="left", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, d["model"])
    vals = np.r_[d.test_top1_accuracy, d.test_macro_f1]
    ax.set_xlim(vals.min() - 0.03, vals.max() + 0.03)
    ax.set_xlabel("score on the 10 student-test trips")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5)
    _title(ax, f"{best.model} has the best student-test top-1 accuracy ({best.test_top1_accuracy:.3f})",
           "Student-test partition, 10 trips (never used for fitting).")
    fig.savefig(out / "p2_test_top1_macro_f1.png"); plt.close(fig)


def plot_log_loss(cmp: pd.DataFrame, out: Path):
    d = cmp.sort_values("test_log_loss")
    best = d.iloc[0]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar(d["model"], d["test_log_loss"], color=_bar_colors(d["model"], best.model), width=0.6)
    _value_labels(ax, bars)
    ax.set_ylabel("log loss (lower is better)")
    _title(ax, f"{best.model} gives the best-calibrated probabilities (log loss {best.test_log_loss:.3f})",
           "Student-test partition. These probabilities become the HMM emissions.")
    fig.savefig(out / "p2_test_log_loss.png"); plt.close(fig)


def plot_inference(cmp: pd.DataFrame, out: Path):
    d = cmp.sort_values("inference_ms_per_1k_points")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar(d["model"], d["inference_ms_per_1k_points"], color=GREY, width=0.6)
    _value_labels(ax, bars, "{:.1f}")
    ax.set_yscale("log")
    ax.set_ylabel("ms per 1,000 GPS points (log scale)")
    slow = d.iloc[-1]
    _title(ax, f"{slow.model} is the slowest at inference ({slow.inference_ms_per_1k_points:.0f} ms / 1k points)",
           "Classifier predict_proba time only, student-test partition, this laptop.")
    fig.savefig(out / "p2_inference_time.png"); plt.close(fig)


def plot_confusions(cmp: pd.DataFrame, out: Path):
    for r in cmp.itertuples():
        cm = np.array([[r.test_tn, r.test_fp], [r.test_fn, r.test_tp]])
        share = cm / cm.sum(axis=1, keepdims=True)  # row-normalised: recall per true class
        fig, ax = plt.subplots(figsize=(3.8, 3.4))
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}\n{share[i, j]:.1%}", ha="center", va="center", fontsize=9,
                    color="white" if share[i, j] > 0.6 else INK)
        ax.set_xticks([0, 1], ["pred wrong arc", "pred correct arc"], fontsize=8)
        ax.set_yticks([0, 1], ["true wrong arc", "true correct arc"], fontsize=8)
        ax.set_title(f"{r.model}: student-test confusion matrix", fontsize=10, color=INK, loc="left")
        fig.savefig(out / f"p2_confusion_{r.model}.png"); plt.close(fig)


def plot_per_model(cmp: pd.DataFrame, folds: pd.DataFrame, diag: Path, out: Path):
    """One 4-panel figure per model (report §3.3 'visualisations for every model'):
    (a) CV per fold, (b) calibration on test, (c) top-1 per test trip, (d) permutation importance.
    The confusion matrix of the same model is p2_confusion_<model>.png."""
    for r in cmp.itertuples():
        name = r.model
        files = {k: diag / f"{name}_{k}.csv" for k in ("calibration", "per_trip", "importance")}
        if not all(f.exists() for f in files.values()):
            print(f"  (skip per-model figure for {name}: run python -m scripts.model_diagnostics)")
            continue
        cal, trips, imp = (pd.read_csv(files[k]) for k in ("calibration", "per_trip", "importance"))
        f = folds[folds.model == name].sort_values("fold")
        fig, axs = plt.subplots(2, 2, figsize=(11, 7.6))
        fig.suptitle(f"{name}, {r.family}: test top-1 {r.test_top1_accuracy:.3f}, test macro-F1 "
                     f"{r.test_macro_f1:.3f}, test log loss {r.test_log_loss:.3f}",
                     x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)

        ax = axs[0, 0]  # (a) stability across folds
        ax.plot(f.fold, f.macro_f1, "o-", color=BLUE, lw=2, label="macro-F1")
        ax.plot(f.fold, f.top1_accuracy, "s-", color=ORANGE, lw=2, label="top-1 accuracy")
        ax.set_xticks(f.fold, [f"fold {k}" for k in f.fold])
        ax.set_ylim(min(f.top1_accuracy.min(), f.macro_f1.min()) - 0.02, max(f.top1_accuracy.max(), f.macro_f1.max()) + 0.02)
        ax.legend(frameon=False, fontsize=8.5, loc="lower right")
        ax.set_title(f"(a) CV on held-out trips: macro-F1 {f.macro_f1.mean():.3f} ± {f.macro_f1.std():.3f}",
                     loc="left", fontsize=10, color=INK)

        ax = axs[0, 1]  # (b) calibration: are the probabilities trustworthy as HMM emissions?
        ax.grid(axis="x")
        c = cal.dropna()
        ax.plot([0, 1], [0, 1], ls="--", color=INK2, lw=1, label="perfect calibration")
        ax.plot(c.mean_pred, c.observed, "o-", color=BLUE, lw=2, label="this model")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("mean predicted probability (bin)"); ax.set_ylabel("observed share of correct arcs")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        ece = float(np.nansum(cal.n * (cal.mean_pred - cal.observed).abs()) / cal.n.sum())
        ax.set_title(f"(b) Calibration on student test (ECE {ece:.3f})", loc="left", fontsize=10, color=INK)

        ax = axs[1, 0]  # (c) per trip: is it good everywhere or carried by a few trips?
        t = trips.sort_values("top1_accuracy")
        yy = np.arange(len(t))
        ax.grid(axis="x"); ax.grid(axis="y", visible=False)
        ax.hlines(yy, t.nearest_arc_top1, t.top1_accuracy, color=GRID, lw=2)
        ax.scatter(t.nearest_arc_top1, yy, s=22, color=GREY, label="nearest-arc rule", zorder=3)
        ax.scatter(t.top1_accuracy, yy, s=30, color=BLUE, label=name, zorder=3)
        ax.set_yticks(yy, [f"{a} ({n:,} pts)" for a, n in zip(t.trip_id.astype(str).str.zfill(8), t.n_points)], fontsize=8)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("top-1 accuracy")
        ax.legend(frameon=False, fontsize=8.5, loc="lower left")
        ax.set_title(f"(c) Top-1 per student-test trip (worst {t.top1_accuracy.min():.3f})", loc="left",
                     fontsize=10, color=INK)

        ax = axs[1, 1]  # (d) which features the model relies on
        im = imp.sort_values("log_loss_increase")
        ax.grid(axis="x"); ax.grid(axis="y", visible=False)
        top = im.feature.iloc[-1]
        ax.barh(im.feature, im.log_loss_increase, xerr=im["std"], color=[BLUE if x == top else GREY for x in im.feature],
                height=0.65, error_kw={"ecolor": INK2, "lw": 0.8})
        ax.axvline(0, color=INK2, lw=0.8)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlabel("increase in log loss when the feature is shuffled")
        ax.set_title(f"(d) Permutation importance: {top} matters most", loc="left", fontsize=10, color=INK)

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out / f"p2_model_{name}.png"); plt.close(fig)


def plot_mlp_history(hist_dir: Path, out: Path):
    f = hist_dir / "mlp_history.csv"
    if not f.exists():
        print(f"  (skip MLP curve: {f} not found)")
        return
    h = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(h.epoch, h.train_log_loss, color=GREY, lw=2, label="train trips")
    ax.plot(h.epoch, h.val_log_loss, color=BLUE, lw=2, label="validation trips (held out)")
    best = h.loc[h.val_log_loss.idxmin()]
    ax.scatter([best.epoch], [best.val_log_loss], color=BLUE, s=30, zorder=3)
    ax.annotate(f"lowest validation loss\nepoch {int(best.epoch)}: {best.val_log_loss:.4f}",
                (best.epoch, best.val_log_loss), xytext=(10, 25), textcoords="offset points", fontsize=8.5,
                color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlabel("epoch"); ax.set_ylabel("log loss")
    ax.legend(frameon=False, fontsize=8.5)
    gap = h.val_log_loss.iloc[-1] - h.train_log_loss.iloc[-1]
    _title(ax, f"MLP training history: final train/validation gap {gap:+.4f}",
           "Train partition split by trip (80/20). MLP is the only model with an epoch-wise history.")
    fig.savefig(out / "p2_mlp_loss_curve.png"); plt.close(fig)


def plot_tuned(res: Path, out: Path):
    f = res / "tuned_vs_untuned.csv"
    if not f.exists():
        return
    d = pd.read_csv(f).set_index("variant")
    metrics = [("cv_macro_f1", "CV macro-F1"), ("test_top1_accuracy", "test top-1"), ("test_macro_f1", "test macro-F1")]
    x = np.arange(len(metrics)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for k, (variant, color) in enumerate([("untuned", GREY), ("tuned", BLUE)]):
        if variant in d.index:
            bars = ax.bar(x + (k - 0.5) * w, [d.loc[variant, m] for m, _ in metrics], w, color=color, label=variant)
            _value_labels(ax, bars, "{:.4f}")
    ax.set_xticks(x, [lbl for _, lbl in metrics])
    ax.set_ylim(0, 1.08)  # bars measured from zero; the exact values are printed on them
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=8.5)
    name = d["model"].iloc[0]
    dt = d.loc["tuned", "test_top1_accuracy"] - d.loc["untuned", "test_top1_accuracy"]
    _title(ax, f"Tuning {name} changed student-test top-1 by {dt * 100:+.2f} points",
           "Same folds, same partitions; values printed on the bars.")
    fig.savefig(out / "p2_tuned_vs_untuned.png"); plt.close(fig)


def plot_optuna(res: Path, out: Path):
    f = res / "optuna" / "trials.csv"
    if not f.exists():
        return
    t = pd.read_csv(f)
    done = t[t.state == "COMPLETE"].sort_values("number")
    pruned = t[t.state == "PRUNED"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    # Zoom on the useful range: from the best value to just above the untuned trial 0.
    base = float(done.loc[done.number == done.number.min(), "value"].iloc[0])
    lo, hi = done.value.min(), base + 2 * (base - done.value.min())
    top = hi - 0.04 * (hi - lo)  # strip near the top for off-scale and pruned trials
    inside, off = done[done.value <= hi], done[done.value > hi]
    ax.scatter(inside.number, inside.value, s=16, color=GREY, label="completed trial", zorder=2)
    if len(off):
        ax.scatter(off.number, np.full(len(off), top), s=24, marker="^", color=GREY, zorder=2,
                   label=f"completed, off scale ({', '.join(f'{v:.3f}' for v in off.value)})")
    if len(pruned):
        ax.scatter(pruned.number, np.full(len(pruned), top), s=12, marker="x", color=INK2, zorder=2,
                   label="pruned (stopped early, worse than median)")
    ax.plot(done.number, done.value.cummin(), color=BLUE, lw=2, drawstyle="steps-post", label="best so far", zorder=3)
    ax.axhline(base, color=INK2, lw=0.8, ls="--")
    ax.text(done.number.max() + 1, base, " untuned\n (trial 0)", ha="left", va="center", fontsize=8, color=INK2)
    ax.set_ylim(lo - 0.15 * (hi - lo), hi)
    ax.set_xlim(-2, len(t) + 6)
    ax.set_xlabel("trial number"); ax.set_ylabel("CV log loss (lower is better)")
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.9))
    _title(ax, f"Optuna TPE: best CV log loss {done.value.min():.4f} at trial {int(done.loc[done.value.idxmin(), 'number'])}",
           f"{len(t)} trials ({len(done)} completed, {len(pruned)} pruned), GroupKFold by trip on train.")
    fig.savefig(out / "p2_optuna_history.png"); plt.close(fig)


def main():
    res = Path(CFG["paths"]["results_dir"])
    out = res / "plots"
    out.mkdir(parents=True, exist_ok=True)
    cmp = pd.read_csv(res / "model_comparison.csv")
    folds = pd.read_csv(res / "cv_folds.csv")
    plot_cv(cmp, folds, out)
    plot_test(cmp, out)
    plot_log_loss(cmp, out)
    plot_inference(cmp, out)
    plot_confusions(cmp, out)
    plot_per_model(cmp, folds, res / "model_diagnostics", out)
    plot_mlp_history(res / "training_history", out)
    plot_tuned(res, out)
    plot_optuna(res, out)
    print(f"plots written to {out}")


if __name__ == "__main__":
    main()
