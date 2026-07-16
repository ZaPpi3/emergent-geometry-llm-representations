"""Compare the real semantic-probe spectrum against the g=0 random-token control."""
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SERIES_REAL = "#2a78d6"   # categorical slot 1
SERIES_CTRL = "#008300"   # categorical slot 2

N_MODES = 15


def main():
    real = np.load("eigenmodes.npz", allow_pickle=True)["eigenvalues"][:N_MODES]
    ctrl = np.load("control_eigenvalues.npz", allow_pickle=True)["eigenvalues"][:N_MODES]
    idx = np.arange(1, N_MODES + 1)

    fig, ax = plt.subplots(figsize=(9, 6.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=INK_MUTED)

    ax.plot(idx, real, color=SERIES_REAL, marker="o", markersize=6,
            linewidth=2, label="Structured semantic probes", zorder=3)
    ax.plot(idx, ctrl, color=SERIES_CTRL, marker="s", markersize=6,
            linewidth=2, label="g=0 control (random vocab tokens)", zorder=3)

    ax.set_xlabel("Eigenmode index", color=INK_PRIMARY)
    ax.set_ylabel("Eigenvalue", color=INK_PRIMARY)
    ax.set_title("Laplacian spectrum: structured probes vs. randomized-token control", color=INK_PRIMARY)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig("spectrum_comparison.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print("Saved spectrum_comparison.png")


if __name__ == "__main__":
    main()
