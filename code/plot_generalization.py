"""Does the rotation-generator effect generalize to other small closed sets,
and does subject-name token length explain the pattern?"""
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
REAL_COLOR = "#2a78d6"
ROT_COLOR = "#eb6834"
POINT_COLOR = "#4a3aa7"

CATEGORIES = ["Days-of-week", "Months", "Playing cards", "Planets", "Zodiac signs"]
REAL_ACC = [1.00, 0.82, 0.82, 0.33, 0.42]
ROT_ACC = [1.00, 0.85, 0.73, 0.17, 0.33]
AVG_TOKENS = [1.00, 1.00, 1.15, 2.00, 2.25]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=SURFACE)

    ax = axes[0]
    x = np.arange(len(CATEGORIES))
    width = 0.35
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors="#898781")
    ax.bar(x - width / 2, REAL_ACC, width, label="Real donor patch", color=REAL_COLOR, zorder=3)
    ax.bar(x + width / 2, ROT_ACC, width, label="Full-rank rotation", color=ROT_COLOR, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES, rotation=20, ha="right", color=INK_PRIMARY, fontsize=9)
    ax.set_ylabel("Patch accuracy", color=INK_PRIMARY)
    ax.set_ylim(0, 1.08)
    ax.set_title("Generalization across categories", color=INK_PRIMARY)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    ax2 = axes[1]
    ax2.set_facecolor(SURFACE)
    for spine in ax2.spines.values():
        spine.set_color(AXIS)
    ax2.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax2.tick_params(colors="#898781")
    ax2.scatter(AVG_TOKENS, REAL_ACC, color=REAL_COLOR, s=80, zorder=3, label="Real donor patch")
    ax2.scatter(AVG_TOKENS, ROT_ACC, color=ROT_COLOR, s=80, zorder=3, marker="s",
                label="Full-rank rotation")
    for i, cat in enumerate(CATEGORIES):
        ax2.annotate(cat, (AVG_TOKENS[i], REAL_ACC[i]), fontsize=8, color=INK_SECONDARY,
                     xytext=(6, 4), textcoords="offset points")
    ax2.set_xlabel("Avg. subject-name token length", color=INK_PRIMARY)
    ax2.set_ylabel("Patch accuracy", color=INK_PRIMARY)
    ax2.set_ylim(0, 1.08)
    ax2.set_title("Accuracy vs. subject-name token length", color=INK_PRIMARY)
    legend2 = ax2.legend(loc="upper right", frameon=False, fontsize=9)
    for text in legend2.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig("generalization_summary.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print("Saved generalization_summary.png")


if __name__ == "__main__":
    main()
