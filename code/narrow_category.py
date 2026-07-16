"""Narrow re-run: does a single cyclic category resolve into its own circle?

Filters activations.npz down to one category, rebuilds the graph + Laplacian
on just those tokens, and plots the eigenmode embedding with the tokens'
natural cyclic order connected — a real circle should trace a closed loop.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from build_graph import cosine_similarity_matrix, standardize, unnormalized_laplacian

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
POINT_COLOR = "#2a78d6"
LOOP_COLOR = "#eb6834"


def main():
    parser = argparse.ArgumentParser(description="Check whether one cyclic category forms its own circle")
    parser.add_argument("--activations", default="activations.npz")
    parser.add_argument("--category", required=True, help="comma-separated category name(s)")
    parser.add_argument("--order", nargs="+", action="append",
                        help="Tokens in natural cyclic order (for drawing the loop); repeat --order for multiple separate cycles")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    wanted = set(args.category.split(","))
    data = np.load(args.activations, allow_pickle=True)
    mask = np.isin(data["categories"], list(wanted))
    vectors = data["vectors"][mask]
    tokens = [str(t) for t in data["tokens"][mask]]
    print(f"{args.category}: {len(tokens)} tokens -> {tokens}")

    vectors = standardize(vectors)
    weights = cosine_similarity_matrix(vectors)
    laplacian = unnormalized_laplacian(weights)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    print("Eigenvalues:", np.round(eigenvalues, 4))

    modes = eigenvectors[:, 1:3]  # lambda_1, lambda_2

    fig, ax = plt.subplots(figsize=(7, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=INK_MUTED)

    loop_colors = ["#eb6834", "#4a3aa7"]
    if args.order:
        for i, order in enumerate(args.order):
            order_idx = [tokens.index(t) for t in order if t in tokens]
            loop_idx = order_idx + [order_idx[0]]
            ax.plot(modes[loop_idx, 0], modes[loop_idx, 1], color=loop_colors[i % len(loop_colors)],
                    linewidth=1.5, zorder=2, label=f"cyclic order: {order[0]}...")

    ax.scatter(modes[:, 0], modes[:, 1], color=POINT_COLOR, s=90,
               edgecolor="white", linewidth=0.6, zorder=3)
    for i, tok in enumerate(tokens):
        ax.annotate(tok, (modes[i, 0], modes[i, 1]), fontsize=9,
                    color=INK_SECONDARY, xytext=(6, 4), textcoords="offset points", zorder=4)

    ax.set_xlabel(r"$\lambda_1$ eigenmode", color=INK_PRIMARY)
    ax.set_ylabel(r"$\lambda_2$ eigenmode", color=INK_PRIMARY)
    ax.set_title(f"{args.category}: eigenmode embedding", color=INK_PRIMARY)
    if args.order:
        legend = ax.legend(loc="best", frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)

    fig.tight_layout()
    out_path = args.out or f"{args.category}_circle.png"
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
