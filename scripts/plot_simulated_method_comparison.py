"""Compare the saved simulated results with CTGAN supplement Table 3."""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "output" / "comparisons" / "simulated_methods"
SOURCE_URL = "https://papers.neurips.cc/paper_files/paper/2019/file/254ed7d2de3b23ab10936522dd547b78-Supplemental.zip"
METRICS = ["l_syn", "l_test", "test_accuracy", "test_macro_f1", "oracle_label_consistency"]
SUPPLEMENT = {
    "grid": {"identity": (-3.06, -3.06), "ctgan": (-5.63, -3.69), "tvae": (-2.86, -11.26)},
    "gridr": {"identity": (-3.06, -3.07), "ctgan": (-8.11, -4.31), "tvae": (-3.41, -3.20)},
    "ring": {"identity": (-1.70, -1.70), "ctgan": (-3.43, -2.19), "tvae": (-1.68, -1.79)},
    "asia": {"identity": (-2.23, -2.24), "ctgan": (-2.56, -2.31), "tvae": (-2.31, -2.27)},
    "alarm": {"identity": (-10.3, -10.3), "ctgan": (-14.2, -12.6), "tvae": (-11.2, -10.7)},
    "child": {"identity": (-12.0, -12.0), "ctgan": (-13.4, -12.7), "tvae": (-12.3, -12.3)},
    "insurance": {"identity": (-12.8, -12.9), "ctgan": (-16.5, -14.8), "tvae": (-14.7, -14.2)},
}
FAMILIES = {"GM": ("grid", "gridr", "ring"), "BN": ("asia", "alarm", "child", "insurance")}
REFERENCES = {
    family: {method: tuple(sum(SUPPLEMENT[d][method][j] for d in datasets) / len(datasets)
                           for j in range(2)) for method in ("identity", "ctgan", "tvae")}
    for family, datasets in FAMILIES.items()
}
REFERENCE_NOTE = "CTGAN mapping: the supplement labels its BN row 'TGAN' and duplicates 'TVAE' in GM; the second GM row is inferred as CTGAN."
SUFFIXES = ["full", "full-categorical", "full-gaussian", "full-pca_gmm", "full-rf", "full-xgb", "full-dnn",
            "categorical", "gaussian", "pca_gmm", "rf", "xgb", "dnn"]
COLORS = {"ctgan": "#008696", "tvae": "#a565cc", "identity": "#263445"}


def read_rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and parts[0] in ("paper", "labeled_extension"):
            if len(parts) != 8:
                raise ValueError(f"Unexpected summary row: {line}")
            rows.append(dict(zip(["benchmark", "method", "family"] + METRICS,
                                 parts[:3] + [float(x) for x in parts[3:]])))
    keys = {(r["benchmark"], r["method"], r["family"]) for r in rows}
    if len(rows) != 58 or len(keys) != 58:
        raise ValueError("Expected the 58 distinct summary rows supplied by the user")
    return rows


def label(benchmark, method, family):
    if benchmark == "paper":
        if method == "identity":
            return "Original data (identity)"
        return method.upper() + (" · generates features only" if family == "GM" else " · generates full table")
    generator, suffix = method.split("-", 1)
    if suffix == "full":
        return generator.upper() + " · generates features + target"
    full = suffix.startswith("full-")
    suffix = suffix.removeprefix("full-")
    names = {"categorical": "Categorical NB", "gaussian": "Gaussian NB", "pca_gmm": "PCA-GMM"}
    return f"{generator.upper()} · {'joint' if full else 'features'} + {names.get(suffix, suffix.upper())} labels"


def ordered(rows, family, utility=False):
    index = {(r["benchmark"], r["method"]): r for r in rows if r["family"] == family}
    keys = [] if utility else [("paper", "identity"), ("paper", "ctgan"), ("paper", "tvae")]
    keys += [("labeled_extension", f"{g}-{s}") for g in ("ctgan", "tvae") for s in SUFFIXES]
    return [index[k] for k in keys]


def panel(ax, records, metric, title, limits, reference=False):
    percent = metric not in ("l_syn", "l_test")
    factor = 100 if percent else 1
    ax.set_title(title, loc="left", fontsize=13, weight="bold", pad=15)
    ax.set_xlim(*limits)
    ax.set_ylim(len(records) - .3, -.8)
    ax.set_yticks(range(len(records)), [label(r["benchmark"], r["method"], r["family"]) for r in records], fontsize=9)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.grid(axis="x", color="#e3e9f0", linewidth=.8)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#8390a1")
    if reference:
        j = METRICS.index(metric)
        for method, style in (("ctgan", "--"), ("tvae", ":"), ("identity", "-.")):
            ax.axvline(REFERENCES[records[0]["family"]][method][j], color=COLORS[method], linestyle=style, alpha=.7)
    previous_group = None
    for i, r in enumerate(records):
        group = (r["benchmark"], r["method"].split("-")[0], "full" in r["method"])
        if previous_group is not None and group != previous_group:
            ax.axhline(i - .5, color="#dce4ee", linewidth=.7)
        previous_group = group
        value = r[metric] * factor
        if not math.isfinite(value):
            ax.text(limits[0] + .04*(limits[1]-limits[0]), i, "Not evaluated", fontsize=8,
                    va="center", color="#8390a1", fontstyle="italic")
            continue
        color = COLORS[r["method"].split("-")[0]]
        ax.scatter(value, i, color=color, s=25, zorder=5)
        offset, align = (5, "left") if value < limits[1] - .085 * (limits[1]-limits[0]) else (-5, "right")
        ax.annotate(f"{value:.1f}%" if percent else f"{value:.3f}", (value, i), xytext=(offset, 0),
                    textcoords="offset points", va="center", ha=align, fontsize=8, color="#263445")
    ax.set_xlabel({"l_syn": "Mean log likelihood L_syn → higher is better",
                   "l_test": "Mean log likelihood L_test → higher is better",
                   "test_accuracy": "Test accuracy (%)", "test_macro_f1": "Test macro F1 (%)",
                   "oracle_label_consistency": "Oracle label consistency (%)"}[metric], fontsize=10)


def save(fig, stem):
    fig.savefig(DEST / f"{stem}.png", dpi=180, facecolor="white")
    fig.savefig(DEST / f"{stem}.svg", facecolor="white")
    plt.close(fig)


def make_figures(rows):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.axisbelow": True})
    fig, axes = plt.subplots(2, 2, figsize=(19, 21))
    fig.subplots_adjust(left=.20, right=.97, top=.90, bottom=.075, wspace=.70, hspace=.20)
    fig.text(.035, .975, "Our synthetic-data pipelines vs. the CTGAN supplement", fontsize=22, weight="bold")
    fig.text(.035, .952, "Audited pre-fix results · seeds 42/43 · references = unweighted means of Supplement Table 3 dataset scores", fontsize=11, color="#61738c")
    handles = [Line2D([], [], color=COLORS[g], linestyle=s, label=f"Supplement {g.upper() if g != 'identity' else 'Identity'} mean")
               for g, s in (("ctgan", "--"), ("tvae", ":"), ("identity", "-."))]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.97, .945), ncol=3, frameon=False)
    for i, family in enumerate(("GM", "BN")):
        for j, metric in enumerate(("l_syn", "l_test")):
            limits = {("GM", "l_syn"): (-6.05, -2.3), ("GM", "l_test"): (-5.75, -2.3),
                      ("BN", "l_syn"): (-12.9, -9.0), ("BN", "l_test"): (-11.55, -9.1)}[(family, metric)]
            panel(axes[i, j], ordered(rows, family), metric, f"{'Gaussian mixtures' if family == 'GM' else 'Bayesian networks'} · {metric}", limits, True)
    fig.text(.035, .044, "References: Supplement Table 3, Xu et al. (NeurIPS 2019). Extended methods were not published; lines provide baseline context.", fontsize=10, color="#61738c")
    fig.text(.035, .028, "GM likelihood scores features only; BN likelihood scores the joint table, so changing labels can change BN likelihood.", fontsize=10, color="#61738c")
    fig.text(.035, .012, REFERENCE_NOTE, fontsize=10, color="#61738c")
    save(fig, "likelihood_comparison")

    fig, axes = plt.subplots(2, 2, figsize=(19, 20))
    fig.subplots_adjust(left=.20, right=.97, top=.925, bottom=.065, wspace=.70, hspace=.20)
    fig.text(.035, .974, "Every label-generation pipeline on held-out oracle test data", fontsize=21, weight="bold")
    fig.text(.035, .950, "Audited pre-fix results · 26 labeled variants · teal = CTGAN · purple = TVAE · downstream evaluator: random forest", fontsize=11, color="#61738c")
    for i, family in enumerate(("GM", "BN")):
        for j, metric in enumerate(("test_accuracy", "test_macro_f1")):
            panel(axes[i, j], ordered(rows, family, True), metric,
                  f"{'Gaussian mixtures' if family == 'GM' else 'Bayesian networks'} · {'accuracy' if j == 0 else 'macro F1'}",
                  (85, 101.5) if family == "GM" else ((74, 90) if j == 0 else (65, 86)))
    fig.text(.035, .036, "RF / XGB / DNN etc. identify the synthetic-label predictor; the final evaluation classifier is always a random forest.", fontsize=10, color="#61738c")
    fig.text(.035, .019, "The supplement has no prediction results for these simulated tasks; no published accuracy/F1 reference exists for these panels.", fontsize=10, color="#61738c")
    save(fig, "prediction_comparison")
    fig, ax = plt.subplots(figsize=(11, 11))
    fig.subplots_adjust(left=.36, right=.96, top=.90, bottom=.09)
    panel(ax, ordered(rows, "GM", True), "oracle_label_consistency", "Gaussian mixtures · agreement with the fixed oracle boundary", (86, 101.5))
    fig.text(.035, .96, "Synthetic-label consistency", fontsize=20, weight="bold")
    fig.text(.035, .035, "Pre-fix results, GM only. Synthetic-feature label agreement; the supplement provides no matching reference.", fontsize=9, color="#61738c")
    save(fig, "oracle_consistency")
    make_combined_figures(rows)


def make_combined_figures(rows):
    for family in FAMILIES:
        fig, axes = plt.subplots(1, 4, figsize=(24, 14))
        fig.subplots_adjust(left=.235, right=.98, top=.84, bottom=.16, wspace=.20)
        title = "Gaussian mixtures" if family == "GM" else "Bayesian networks"
        fig.text(.03, .96, f"{title}: distribution fit and prediction performance", fontsize=23, weight="bold")
        fig.text(.03, .933, "Audited pre-fix results · means over datasets and seeds 42/43 · points = ours · vertical lines = supplement likelihood averages",
                 fontsize=11, color="#61738c")
        handles = [Line2D([], [], color=COLORS[g], linestyle=s, label=f"Supplement {g.upper() if g != 'identity' else 'Identity'}")
                   for g, s in (("ctgan", "--"), ("tvae", ":"), ("identity", "-."))]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.98, .915), ncol=3, frameon=False)
        limits = ((-6.05, -2.3), (-5.75, -2.3), (85, 101.5), (85, 101.5)) if family == "GM" else (
            (-12.9, -9.0), (-11.55, -9.1), (74, 90), (65, 86))
        titles = ("L_syn\nOriginal model scores synthetic rows", "L_test\nFitted model scores test rows",
                  "Accuracy\nRF predicts held-out targets", "Macro F1\nRF predicts held-out targets")
        for j, metric in enumerate(METRICS[:4]):
            panel(axes[j], ordered(rows, family), metric, titles[j], limits[j], reference=j < 2)
            axes[j].title.set_fontsize(10)
            if j:
                axes[j].tick_params(axis="y", labelleft=False)
        fig.text(.03, .115, "L_syn = average log p_original(synthetic row). L_test = average log p_fitted-on-synthetic(test row). Higher is better for both.",
                 fontsize=11, color="#61738c")
        fig.text(.03, .090, "Accuracy = correct predictions / test rows. Macro F1 = unweighted mean of per-class F1. Evaluation RF is trained on synthetic features + labels.",
                 fontsize=11, color="#61738c")
        detail = ("GM labels are our added rule: feature_1 > 1.5 × feature_0 + 0.8. Likelihood uses only the two features; it ignores those labels."
                  if family == "GM" else "BN target columns: Asia dysp, Alarm BP, Child Disease, Insurance Accident. Likelihood includes all columns, including the target.")
        fig.text(.03, .065, detail, fontsize=11, color="#61738c")
        fig.text(.03, .040, "RF/XGB/DNN/NB labelers learn from original training labels. Original paper baselines have no saved prediction evaluation; supplement has none for these tasks.",
                 fontsize=10, color="#61738c")
        fig.text(.03, .017, REFERENCE_NOTE, fontsize=9, color="#61738c")
        save(fig, f"{family.lower()}_all_metrics_comparison")


def export(rows):
    fields = ["benchmark", "method", "family"] + METRICS + ["supplement_l_syn", "supplement_l_test", "delta_l_syn", "delta_l_test"]
    with (DEST / "all_methods_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            generator = r["method"].split("-")[0]
            a, b = REFERENCES[r["family"]][generator]
            writer.writerow({**r, "supplement_l_syn": a, "supplement_l_test": b,
                             "delta_l_syn": round(r["l_syn"]-a, 6), "delta_l_test": round(r["l_test"]-b, 6)})
    lines = ["# Comprehensive simulated-method comparison", "", f"Published references: [CTGAN supplement, Table 3]({SOURCE_URL}).", "",
             "All 58 pasted rows are preserved. Deltas compare each variant with its generator's published baseline; positive means higher likelihood. These are descriptive differences, not statistical significance or an exact reproduction claim.", "",
             "## How to read method names", "", "`ctgan-rf` generates X-only features and predicts their targets with RF; `ctgan-full-rf` generates the table with the target present, then replaces its target with RF predictions. The same convention applies to TVAE, XGB, DNN, both naive Bayes methods and PCA-GMM. `*-full` retains the generated target. Every labeled variant is evaluated by a fresh random forest trained on its synthetic table.", "",
             "GM likelihood ignores labels; repeated scores across labelers sharing features are expected. BN likelihood includes the target, so relabeling changes the joint distribution. Oracle consistency is defined only for GM.", "",
             "## How the metrics are calculated", "",
             "X means the feature columns; CTGAN itself generates X. For GM, CTGAN trained on the two original continuous columns generates two-column rows. Joint generation trains a separate CTGAN on those columns plus our added discrete target, then generates all three columns. For BN, the target already exists in the original table, so the baseline and generated-target variants reuse the same full-table sample.", "",
             "L_syn is the mean log probability/density of synthetic rows under the known original oracle. L_test fits a density/probability model to the synthetic rows, then averages its log probability/density on the independent original test rows. GM refitting uses a diagonal-covariance Gaussian mixture with the original number of components; BN refitting keeps the original graph and estimates conditional probability tables. BN scoring uses log(p + 1e-8). GM scores features only; BN scores the complete table.", "",
             "For prediction, GM receives an added target: label = 1 if feature_1 > 1.5 * feature_0 + 0.8, else 0. BN targets are dysp (Asia), BP (Alarm), Disease (Child), and Accident (Insurance). A fresh 100-tree random forest trains on synthetic features and targets and predicts targets on the independent original test set. Accuracy is the fraction correct. Macro F1 is the unweighted average of per-class F1, with zero_division=0. Labeler models learn using original training labels; they are distinct from the final evaluation RF. These prediction tasks are our extension, not the paper's simulated benchmark.", "",
             "The GM and BN combined figures show likelihood, accuracy and macro F1 on aligned method rows. Not evaluated means that no prediction result was saved for that original paper-baseline row. All family figures average datasets equally and average the two seeds, rather than pooling their classification predictions.", "",
             "The pasted summary alone does not identify coverage or variability. The subsequent server audit verified all seven datasets, seeds 42 and 43, 10,000 train/test/synthetic rows, and recorded settings of 300 epochs on CUDA. The full raw results were saved separately in audit/remote_simulated_methods_per_run.csv. The paper's real-data F1 values do not supply a baseline for the added simulated prediction tasks.", "",
             "## Supplement-only references", "",
             "All chart references and deltas below use unweighted dataset averages calculated from the rounded numbers in Supplement Table 3. Main-paper averages are not used. Our observations are the audited pre-fix run, not a new post-fix training run.", "",
             REFERENCE_NOTE, "",
             "Main Table 2 reports TVAE BN L_syn=-6.76 and L_test=-9.59. Supplement means are -10.1275 and -9.8675, close to ours (-10.149895, -9.874837). This is an unresolved internal publication inconsistency.", "",
             "| Family | Model | Supplement mean L_syn | Supplement mean L_test |", "|---|---|---:|---:|"]
    for family, methods in REFERENCES.items():
        for method, (a, b) in methods.items():
            lines.append(f"| {family} | {method} | {a:.6f} | {b:.6f} |")
    lines.append("")
    for family in ("GM", "BN"):
        lines += [f"## {family}: all supplied methods", "", "| Method | L_syn | L_test | Δ L_syn vs supplement | Δ L_test vs supplement | Accuracy % | Macro F1 % | Oracle consistency % |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in ordered(rows, family):
            a, b = REFERENCES[family][r["method"].split("-")[0]]
            pct = [f"{r[m]*100:.3f}" if math.isfinite(r[m]) else "—" for m in METRICS[2:]]
            lines.append(f"| {r['method']} ({r['benchmark']}) | {r['l_syn']:.6f} | {r['l_test']:.6f} | {r['l_syn']-a:+.6f} | {r['l_test']-b:+.6f} | " + " | ".join(pct) + " |")
        lines.append("")
    (DEST / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def make_dataset_comparison(path):
    with path.open(newline="", encoding="utf-8") as f:
        runs = [r for r in csv.DictReader(f) if r["benchmark"] == "paper"]
    fig, axes = plt.subplots(7, 2, figsize=(16, 20))
    fig.subplots_adjust(left=.12, right=.95, top=.91, bottom=.07, hspace=.75, wspace=.35)
    fig.text(.04, .973, "Supplement vs. our results, dataset by dataset", fontsize=22, weight="bold")
    fig.text(.04, .950, "Supplement Table 3 · our points = mean of seeds 42/43 from the audited pre-fix run · higher is better", fontsize=11, color="#61738c")
    fig.legend(handles=[Line2D([], [], color="#61738c", marker="o", markerfacecolor="white", linestyle="", label="Supplement"),
                        Line2D([], [], color="#61738c", marker="o", linestyle="", label="Ours")],
               loc="upper right", bbox_to_anchor=(.95, .935), ncol=2, frameon=False)
    comparison = []
    for i, dataset in enumerate(SUPPLEMENT):
        for j, metric in enumerate(("l_syn", "l_test")):
            ax = axes[i, j]
            values = []
            for k, method in enumerate(("identity", "ctgan", "tvae")):
                selected = [r for r in runs if r["dataset"] == dataset and r["method"] == method]
                if len(selected) != 2 or {r["seed"] for r in selected} != {"42", "43"}:
                    raise ValueError(f"Expected both seeds for {dataset}/{method}")
                ours = sum(float(r[metric]) for r in selected) / 2
                published = SUPPLEMENT[dataset][method][j]
                values.extend((ours, published))
                color = COLORS[method]
                ax.plot([published, ours], [k, k], color=color, alpha=.45)
                ax.scatter(published, k, s=55, facecolors="white", edgecolors=color, linewidths=1.5, zorder=4)
                ax.scatter(ours, k, s=30, color=color, zorder=5)
                for value, shift, prefix in ((published, 10, "S"), (ours, -13, "O")):
                    ax.annotate(f"{prefix}: {value:.3f}", (value, k), xytext=(0, shift),
                                textcoords="offset points", ha="center", fontsize=8, color=color)
                comparison.append({"dataset": dataset, "method": method, "metric": metric,
                                   "supplement": published, "ours_mean": ours, "delta": ours-published})
            width = max(values)-min(values)
            padding = max(.15, width*.17)
            ax.set_xlim(min(values)-padding, max(values)+padding)
            ax.set_ylim(2.6, -.6)
            ax.set_yticks(range(3), ["Original data", "CTGAN", "TVAE"])
            ax.set_title(f"{dataset.upper()} · {metric}", loc="left", fontsize=12, weight="bold", pad=12)
            ax.grid(axis="x", color="#e3e9f0")
            ax.tick_params(axis="y", length=0)
            for spine in ("left", "right", "top"):
                ax.spines[spine].set_visible(False)
    fig.text(.04, .038, REFERENCE_NOTE, fontsize=9, color="#61738c")
    fig.text(.04, .020, "Reference values are rounded published scores. Differences are descriptive; two seeds do not establish statistical significance.", fontsize=9, color="#61738c")
    save(fig, "dataset_likelihood_comparison")
    with (DEST / "dataset_likelihood_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--per-run", type=Path, default=ROOT / "audit" / "remote_simulated_methods_per_run.csv")
    args = parser.parse_args()
    rows = read_rows(args.source)
    DEST.mkdir(parents=True, exist_ok=True)
    export(rows)
    make_figures(rows)
    make_dataset_comparison(args.per_run)
    print(f"Exported {len(rows)} summary rows and six charts to {DEST}")
