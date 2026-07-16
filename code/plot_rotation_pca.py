"""Plot the position-4, layer-12 PCA embedding used for the rotation-patch test."""
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
POINT_COLOR = "#2a78d6"
LOOP_COLOR = "#eb6834"

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def plot_one(ax, coords, items, order, title, variance):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors="#898781")

    order_idx = [items.index(t) for t in order]
    loop_idx = order_idx + [order_idx[0]]
    ax.plot(coords[loop_idx, 0], coords[loop_idx, 1], color=LOOP_COLOR, linewidth=1.5, zorder=2)
    ax.scatter(coords[:, 0], coords[:, 1], color=POINT_COLOR, s=70, edgecolor="white",
               linewidth=0.5, zorder=3)
    for i, label in enumerate(items):
        ax.annotate(label, (coords[i, 0], coords[i, 1]), fontsize=8, color=INK_SECONDARY,
                    xytext=(5, 3), textcoords="offset points", zorder=4)
    ax.set_xlabel("PC1", color=INK_PRIMARY)
    ax.set_ylabel("PC2", color=INK_PRIMARY)
    ax.set_title(f"{title}\n(explained variance: {variance[0]:.2f} + {variance[1]:.2f} = "
                 f"{sum(variance):.2f})", color=INK_PRIMARY)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor=SURFACE)
    d = np.load("rotation_days_of_week_pca.npz", allow_pickle=True)
    m = np.load("rotation_months_pca.npz", allow_pickle=True)
    plot_one(axes[0], d["coords"], [str(x) for x in d["items"]], DAYS,
              "Days-of-week: subject position, layer 12", d["explained_variance"])
    plot_one(axes[1], m["coords"], [str(x) for x in m["items"]], MONTHS,
              "Months: subject position, layer 12", m["explained_variance"])
    fig.tight_layout()
    fig.savefig("rotation_pca.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print("Saved rotation_pca.png")


if __name__ == "__main__":
    main()
