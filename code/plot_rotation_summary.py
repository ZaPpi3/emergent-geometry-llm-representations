"""Summary bar chart: patch accuracy across all tested intervention conditions."""
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
DAYS_COLOR = "#2a78d6"
MONTHS_COLOR = "#008300"

CONDITIONS = ["Real donor\nvector", "2D rotation\n(top-2 PCs)", "2D rotation\noff-lattice ctrl",
              "Full-rank rotation\n(n-1 PCs)", "Full-rank rotation\nfractional-power ctrl"]
DAYS_ACC = [1.000, 0.086, 0.086, 1.000, 0.143]
MONTHS_ACC = [0.817, 0.000, 0.000, 0.850, 0.083]


def main():
    x = np.arange(len(CONDITIONS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors="#898781")

    ax.bar(x - width / 2, DAYS_ACC, width, label="Days-of-week", color=DAYS_COLOR, zorder=3)
    ax.bar(x + width / 2, MONTHS_ACC, width, label="Months", color=MONTHS_COLOR, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS, color=INK_PRIMARY, fontsize=9)
    ax.set_ylabel("Patch accuracy", color=INK_PRIMARY)
    ax.set_ylim(0, 1.08)
    ax.set_title("Causal patching: every condition tested, layer 12, subject-token position",
                 color=INK_PRIMARY)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    for i, (d, m) in enumerate(zip(DAYS_ACC, MONTHS_ACC)):
        ax.annotate(f"{d:.2f}", (i - width / 2, d), ha="center", va="bottom", fontsize=8.5,
                    color=INK_SECONDARY, xytext=(0, 3), textcoords="offset points")
        ax.annotate(f"{m:.2f}", (i + width / 2, m), ha="center", va="bottom", fontsize=8.5,
                    color=INK_SECONDARY, xytext=(0, 3), textcoords="offset points")

    fig.tight_layout()
    fig.savefig("rotation_summary.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print("Saved rotation_summary.png")


if __name__ == "__main__":
    main()
