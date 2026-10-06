"""Plot existing corrected seed-42 samples; run on the research server."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path.cwd()
SOURCE = ROOT / "data/corrected_v2_seed42_mnist28_news/mnist28/seed_42"
DEST = ROOT / "output/comparisons/mnist28_digit_samples_seed42"
METHODS = [
    ("Original", "real_train_onehot"),
    ("CTGAN", "ctgan_full_generated_100k"),
    ("CTGAN + XGB", "ctgan_xonly_xgb_100k"),
    ("CTGAN + DNN", "ctgan_xonly_dnn_100k"),
]


def select_samples(path):
    # Independent random priorities give a uniform selection within each label.
    rng = np.random.default_rng(42)
    best = np.full(10, np.inf)
    images = np.empty((10, 28, 28), dtype=np.float32)
    indices = np.full(10, -1, dtype=np.int64)
    counts = np.zeros(10, dtype=np.int64)
    offset = 0
    pixels = [str(i) for i in range(784)]
    for chunk in pd.read_csv(path, chunksize=5000, dtype=np.float32):
        assert set(chunk.columns) == set(pixels + ["label"]), path
        labels = chunk["label"].to_numpy()
        assert np.isin(labels, np.arange(10)).all(), path
        values = chunk[pixels].to_numpy()
        assert np.isfinite(values).all() and np.isin(values, [0, 1]).all(), path
        priorities = rng.random(len(chunk))
        for digit in range(10):
            positions = np.flatnonzero(labels == digit)
            counts[digit] += len(positions)
            if len(positions):
                position = positions[np.argmin(priorities[positions])]
                if priorities[position] < best[digit]:
                    best[digit] = priorities[position]
                    indices[digit] = offset + position
                    images[digit] = values[position].reshape(28, 28)
        offset += len(chunk)
    assert (indices >= 0).all(), f"Missing label in {path}"
    if "100k" in path.stem:
        quality = json.loads(path.with_suffix(".quality.json").read_text())
        assert offset == quality["rows"] == 100000, path
        assert {str(i): int(n) for i, n in enumerate(counts)} == quality["synthetic_label_counts"], path
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return images, {
        "source": str(path), "sha256": digest.hexdigest(), "rows": offset,
        "label_counts": counts.tolist(), "selected_zero_based_rows_by_digit": indices.tolist(),
    }


def main():
    # A new output folder prevents overwriting artifacts from another chat.
    DEST.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(4, 10, figsize=(14, 6.2))
    records = []
    for row, (name, suffix) in enumerate(METHODS):
        images, record = select_samples(SOURCE / f"mnist28_seed42_{suffix}.csv")
        records.append({"method": name, **record})
        for digit in range(10):
            ax = axes[row, digit]
            ax.imshow(images[digit], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(str(digit), fontsize=15, pad=10)
            if digit == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center", labelpad=15, fontsize=12)
        print(f"{name}: {record['rows']} rows; all labels 0–9 verified", flush=True)
    fig.suptitle("MNIST28: original and CTGAN samples by label", fontsize=19, y=0.97)
    fig.text(0.55, 0.89, "Digit label", ha="center", fontsize=11, color="#555555")
    fig.text(0.5, 0.045,
             "Corrected seed 42 · one random sample per label · binary pixels\n"
             "CTGAN uses generated labels; XGB / DNN predict labels on features-only CTGAN samples.",
             ha="center", fontsize=10, color="#555555")
    fig.subplots_adjust(left=0.17, right=0.98, top=0.83, bottom=0.15, wspace=0.12, hspace=0.18)
    for extension in ("png", "pdf"):
        fig.savefig(DEST / f"mnist28_comparison.{extension}", dpi=220, facecolor="white")
    plt.close(fig)
    (DEST / "provenance.json").write_text(json.dumps({
        "namespace": "corrected_v2_seed42_mnist28_news", "seed": 42,
        "selection": "uniform random priority per row within each label; RNG seed 42 per source",
        "pixel_order": "numeric columns 0 through 783, row-major 28x28",
        "methods": records,
    }, indent=2) + "\n")
    print(DEST, flush=True)


if __name__ == "__main__":
    main()
