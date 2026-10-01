"""Four aligned panels; seed markers show variation without hiding it in a mean."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = {"original": "#8594aa", "baseline": "#263445",
          "joint": "#008696", "features": "#a565cc"}


def group(method):
    if method == "original":
        return "original"
    return method.split("-")[1] if "-" in method else "baseline"


def label(method):
    if method == "original":
        return "Original (process sample)"
    if "-" not in method:
        return method.upper() + " (generated target)"
    generator, source, labeler = method.split("-")
    names = {"gaussian": "Gaussian NB", "categorical": "Categorical NB", "pca_gmm": "PCA-GMM"}
    return f"{generator.upper()} · {source} + {names.get(labeler, labeler.upper())}"


def plot_results(results, noise, output, epochs):
    metrics = ["joint_log_likelihood", "accuracy", "macro_f1", "h_star_agreement"]
    titles = ["Joint log likelihood", "Test accuracy", "Test macro F1", "Agreement with clean rule h*"]
    methods = list(results["method"].unique())
    fig, axes = plt.subplots(1, 4, figsize=(20, max(7, len(methods) * .4 + 3)))
    fig.subplots_adjust(left=.18, right=.98, top=.77, bottom=.17, wspace=.3)
    fig.suptitle("Known-distribution simulated benchmark", fontsize=18, x=.03, ha="left")
    for j, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[j]
        for generator, style in (("ctgan", "--"), ("tvae", ":")):
            if generator in methods:
                reference = results.loc[results.method == generator, metric].mean() * (1 if j == 0 else 100)
                ax.axvline(reference, color=COLORS["baseline"], linestyle=style, linewidth=1.3, alpha=.65)
        previous = None
        for i, method in enumerate(methods):
            color = COLORS[group(method)]
            section = (method.split("-")[0], group(method))
            if previous is not None and section != previous:
                ax.axhline(i-.5, color="#e3e9f0", linewidth=.8)
            previous = section
            values = results.loc[results.method == method, metric].to_numpy() * (1 if j == 0 else 100)
            ax.scatter(values, [i] * len(values), s=45, facecolors="none", edgecolors=color, zorder=3)
            ax.scatter(values.mean(), i, s=22, color=color, zorder=4)
            ax.annotate(f"{values.mean():.3f}" if j == 0 else f"{values.mean():.1f}%",
                        (values.mean(), i), xytext=(5, 6), textcoords="offset points", fontsize=8)
        ax.set_ylim(len(methods) - .5, -.5)
        ax.set_yticks(range(len(methods)), [label(m) for m in methods] if j == 0 else [""] * len(methods))
        if j == 0:
            for tick, method in zip(ax.get_yticklabels(), methods):
                tick.set_color(COLORS[group(method)])
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(axis="x", color="#e3e9f0")
        ax.tick_params(axis="y", length=0)
        for spine in ("left", "right", "top"):
            ax.spines[spine].set_visible(False)
        ax.set_xlabel("Mean log density → higher is better" if j == 0 else "Percent → higher is better", fontsize=9)
        if j:
            ax.set_xlim(0, 107)
        else:
            ax.margins(x=.2)
    axes[1].axvline(100 * (1-noise), color="#c47820", linestyle="-.", linewidth=1.3)
    color_handles = [Line2D([], [], color=color, marker="o", linestyle="", label=name)
                     for color, name in ((COLORS["original"], "Process-sampled baseline"),
                                         (COLORS["baseline"], "CTGAN / TVAE baselines"),
                                         (COLORS["joint"], "Joint generation + new labels"),
                                         (COLORS["features"], "Features-only generation + labels"))]
    line_handles = [Line2D([], [], color=COLORS["baseline"], linestyle=style, label=generator.upper() + " baseline mean")
                    for generator, style in (("ctgan", "--"), ("tvae", ":")) if generator in methods]
    line_handles.append(Line2D([], [], color="#c47820", linestyle="-.", label=f"Best expected accuracy: {1-noise:.0%}"))
    fig.legend(handles=color_handles, loc="upper left", bbox_to_anchor=(.03, .89), ncol=4, frameon=False, fontsize=10)
    fig.legend(handles=line_handles, loc="upper left", bbox_to_anchor=(.03, .845), ncol=3, frameon=False, fontsize=10)
    fig.text(.03, .94, f"Generator epochs: {epochs} · seeds: {results.seed.nunique()} · target flip probability {noise:.0%}", fontsize=10)
    fig.text(.03, .915, "All prediction metrics use fresh DNNs. Open circles = seeds; filled circles = means. Lines = CTGAN/TVAE baseline means.", fontsize=10)
    fig.text(.03, .10, "Original: DNN trains on original process samples. Synthetic methods: DNN trains on generated labeled rows. All test on independent process samples.", fontsize=9)
    fig.text(.03, .065, "Likelihood uses the fixed, known joint formula on each table, including the original sample. RF appears only in RF relabeling variants.", fontsize=9)
    fig.text(.03, .03, f"Best expected accuracy = 1 − noise = {1-noise:.0%}. Finite test scores can fluctuate above it; agreement with h* excludes label noise.", fontsize=9)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"comparison.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)
