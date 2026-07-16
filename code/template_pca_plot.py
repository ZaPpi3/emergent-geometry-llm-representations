"""PCA the template-based activations and check whether calendar order traces a loop."""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
POINT_COLOR = "#2a78d6"
LOOP_COLOR = "#eb6834"


def plot_category(vectors, labels, order, title, out_path):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    print(f"{title}: explained variance ratio {pca.explained_variance_ratio_.round(3)}")

    fig, ax = plt.subplots(figsize=(7, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors="#898781")

    order_idx = [labels.index(t) for t in order]
    loop_idx = order_idx + [order_idx[0]]
    ax.plot(coords[loop_idx, 0], coords[loop_idx, 1], color=LOOP_COLOR, linewidth=1.5, zorder=2)
    ax.scatter(coords[:, 0], coords[:, 1], color=POINT_COLOR, s=90, edgecolor="white",
               linewidth=0.6, zorder=3)
    for i, label in enumerate(labels):
        ax.annotate(label, (coords[i, 0], coords[i, 1]), fontsize=9, color=INK_SECONDARY,
                    xytext=(6, 4), textcoords="offset points", zorder=4)

    ax.set_xlabel("PC1", color=INK_PRIMARY)
    ax.set_ylabel("PC2", color=INK_PRIMARY)
    ax.set_title(title, color=INK_PRIMARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", default="template_activations.npz")
    args = parser.parse_args()

    data = np.load(args.activations, allow_pickle=True)
    vectors = data["vectors"]
    labels = [str(l) for l in data["labels"]]
    categories = data["categories"]

    days_mask = categories == "days_of_week"
    months_mask = categories == "months"

    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    months_order = ["January", "February", "March", "April", "May", "June", "July", "August",
                     "September", "October", "November", "December"]

    plot_category(vectors[days_mask], [l for l, m in zip(labels, days_mask) if m],
                  days_order, "Days-of-week template activations (PCA)", "days_template_pca.png")
    plot_category(vectors[months_mask], [l for l, m in zip(labels, months_mask) if m],
                  months_order, "Months template activations (PCA)", "months_template_pca.png")


if __name__ == "__main__":
    main()
