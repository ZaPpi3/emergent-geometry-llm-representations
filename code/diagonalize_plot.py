"""Phase 3: Exact-Diagonalisation & Manifold Mapping.

Diagonalises the graph Laplacian from Phase 2 and plots the lowest-lying
non-trivial eigenmodes as spatial coordinates, checking whether they trace
smooth low-dimensional geometry (circles, lines) matching the probe categories.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

# Every point is also directly labeled with its own token text, so color/marker
# here are a secondary grouping aid, not the sole identity channel: with up to
# ~25 categories, no color assignment can stay CVD-safe on every pairwise
# comparison, but the label always disambiguates unambiguously regardless.
PALETTE_COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a",
                   "#eb6834", "#4a3aa7", "#e34948"]
PALETTE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def build_categorical_map(categories):
    unique_cats = sorted(set(str(c) for c in categories))
    mapping = {}
    for i, cat in enumerate(unique_cats):
        color = PALETTE_COLORS[i % len(PALETTE_COLORS)]
        marker = PALETTE_MARKERS[(i // len(PALETTE_COLORS)) % len(PALETTE_MARKERS)]
        mapping[cat] = (color, marker)
    return mapping

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"


def scatter_categories(ax, modes, tokens, categories, dims, categorical_map, label_points=True):
    seen = set()
    for i, tok in enumerate(tokens):
        cat = str(categories[i])
        color, marker = categorical_map.get(cat, ("#898781", "o"))
        label = cat if cat not in seen else None
        seen.add(cat)
        point = [modes[i, d] for d in dims]
        ax.scatter(*point, color=color, marker=marker, s=60,
                   edgecolor="white", linewidth=0.5, zorder=3, label=label)
        if label_points and len(dims) == 2:
            ax.annotate(str(tok), (point[0], point[1]), fontsize=6.5,
                        color=INK_SECONDARY, xytext=(4, 3),
                        textcoords="offset points", zorder=4)


def main():
    parser = argparse.ArgumentParser(description="Phase 3: diagonalize Laplacian and plot eigenmodes")
    parser.add_argument("--graph", default="graph.npz")
    parser.add_argument("--out-prefix", default="eigenmodes")
    parser.add_argument("--n-modes", type=int, default=3)
    parser.add_argument("--title", default="Graph Laplacian low-lying eigenmodes (Qwen2.5-0.5B, layer 12)")
    args = parser.parse_args()

    data = np.load(args.graph, allow_pickle=True)
    laplacian = data["laplacian"]
    tokens = data["tokens"]
    categories = data["categories"]

    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    print("Lowest 10 eigenvalues:", np.round(eigenvalues[:10], 5))

    # skip index 0: the near-zero, constant mode for a connected graph carries no structure
    modes = eigenvectors[:, 1:1 + args.n_modes]
    np.savez(f"{args.out_prefix}.npz", eigenvalues=eigenvalues, modes=modes,
              tokens=tokens, categories=categories)

    categorical_map = build_categorical_map(categories)
    n_cats = len(categorical_map)
    legend_cols = 1 if n_cats <= 10 else 2
    legend_fontsize = 8 if n_cats <= 10 else 6.5

    fig, ax = plt.subplots(figsize=(12.5, 8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=INK_MUTED)

    scatter_categories(ax, modes, tokens, categories, dims=(0, 1), categorical_map=categorical_map)

    ax.set_xlabel(r"$\lambda_1$ eigenmode", color=INK_PRIMARY)
    ax.set_ylabel(r"$\lambda_2$ eigenmode", color=INK_PRIMARY)
    ax.set_title(args.title, color=INK_PRIMARY)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
                        fontsize=legend_fontsize, ncol=legend_cols)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_2d.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {args.out_prefix}_2d.png")

    if args.n_modes >= 3:
        fig3d = plt.figure(figsize=(12.5, 8), facecolor=SURFACE)
        ax3 = fig3d.add_subplot(projection="3d", facecolor=SURFACE)
        scatter_categories(ax3, modes, tokens, categories, dims=(0, 1, 2),
                           categorical_map=categorical_map, label_points=False)
        ax3.set_xlabel(r"$\lambda_1$")
        ax3.set_ylabel(r"$\lambda_2$")
        ax3.set_zlabel(r"$\lambda_3$")
        ax3.set_title("3D eigenmode embedding", color=INK_PRIMARY)
        ax3.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), frameon=False,
                   fontsize=legend_fontsize, ncol=legend_cols)
        fig3d.tight_layout()
        fig3d.savefig(f"{args.out_prefix}_3d.png", dpi=160, facecolor=SURFACE)
        plt.close(fig3d)
        print(f"Saved {args.out_prefix}_3d.png")


if __name__ == "__main__":
    main()
