"""Plot the causal-tracing layer sweep: patch accuracy vs. layer depth."""
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
REAL_COLOR = "#2a78d6"
CONTROL_COLOR = "#e34948"


def plot_one(ax, real, control, title):
    layers = np.arange(len(real))
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors="#898781")
    ax.plot(layers, real, color=REAL_COLOR, marker="o", markersize=5, linewidth=2,
            label="Real donor patch", zorder=3)
    ax.plot(layers, control, color=CONTROL_COLOR, marker="s", markersize=5, linewidth=2,
            label="Random-direction control", zorder=3)
    ax.set_xlabel("Patched layer index", color=INK_PRIMARY)
    ax.set_ylabel("Patch accuracy", color=INK_PRIMARY)
    ax.set_title(title, color=INK_PRIMARY)
    ax.set_ylim(-0.05, 1.05)
    legend = ax.legend(loc="center left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)


def main():
    data = np.load("causal_trace.npz")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=SURFACE)
    plot_one(axes[0], data["days_real"], data["days_control"],
              "Days-of-week: subject-position patch accuracy")
    plot_one(axes[1], data["months_real"], data["months_control"],
              "Months: subject-position patch accuracy")
    fig.tight_layout()
    fig.savefig("causal_trace.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print("Saved causal_trace.png")


if __name__ == "__main__":
    main()
