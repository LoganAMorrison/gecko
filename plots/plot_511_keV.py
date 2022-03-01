"""
Script for generating a plot of GECCO's capabilities of detecting a 511 keV
line.
"""
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import numpy as np

THIS_DIR = Path(__file__).parent

COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#999999",
]

mpl.rcParams["axes.prop_cycle"] = cycler(color=COLORS)

Flux = namedtuple("Flux", ["nevents", "uncertainty", "source_name", "plot_color"])

PHI_MW = Flux(9.6e-4, 0.7e-4, "MW", COLORS[0])
PHI_M31 = Flux(5.76e-6, 4.71e-6, "M31", COLORS[1])
PHI_M33 = Flux(8.09e-8, 3.58e-8, "M33", COLORS[2])
PHI_DRACO = Flux(1.49e-8, 0.62e-8, "Draco", COLORS[3])
PHI_URSA_MINOR = Flux(3.85e-8, 1.44e-8, "Ursa Minor", COLORS[4])
PHI_FORNAX_CL = Flux(7.98e-7, 4.55e-7, "Fornax Cl.", COLORS[6])
PHI_COMA_CL = Flux(1.86e-7, 1.70e-7, "Coma Cl.", COLORS[7])

PHI_GECCO_BC = 7.4e-8
PHI_GECCO_WC = 3.2e-7

GOLD_RATIO = (1.0 + np.sqrt(5)) / 2.0


def make_plot_point_src_vs_dist(ax):
    GECCO_BC = 7.4e-8
    GECCO_WC = 3.2e-7

    PHI_511 = 3e-4
    N_MSP = (9.2e3, 3.1e3)

    ds = np.logspace(-2, 2, 100)
    num_bc = PHI_511 / GECCO_BC * (8.12 / ds) ** 2
    num_wc = PHI_511 / GECCO_WC * (8.12 / ds) ** 2

    ax.fill_between(
        ds,
        num_bc,
        num_wc,
        alpha=0.5,
        color=COLORS[0],
        label="GECCO (best-case)",
    )
    ax.fill_between(
        ds, num_wc, alpha=0.5, color=COLORS[1], label="GECCO (conservative)"
    )

    ax.plot(ds, [N_MSP[0] for _ in ds], ls="-", lw=2, c="k")
    ax.fill_between(
        ds,
        [N_MSP[0] + N_MSP[1] for _ in ds],
        [N_MSP[0] - N_MSP[1] for _ in ds],
        # color="mediumorchid",
        color=COLORS[3],
        alpha=0.5,
    )

    hline_text_x = 2

    ax.text(hline_text_x, 1.4e4, "MSP", fontsize=12)

    ax.plot(ds, [3000 for _ in ds], ls="-", lw=1.5, c="k")
    ax.text(hline_text_x, 3200, "LMXB", fontsize=12)

    N_WR = (1900, 250)
    ax.plot(ds, [N_WR[0] for _ in ds], ls="-", lw=1.5, c="k")
    ax.fill_between(
        ds,
        [N_WR[0] + N_WR[1] for _ in ds],
        [N_WR[0] - N_WR[1] for _ in ds],
        # color="mediumorchid",
        color=COLORS[3],
        alpha=0.5,
    )
    ax.text(hline_text_x, 1100, "Wolf-Rayet", fontsize=12)

    vline_text_y = 1.6e4

    # Wolf-Rayet
    D_WR = 0.350
    ax.vlines(D_WR, Y_MIN, Y_MAX, colors="k", linestyles="--")
    ax.text(D_WR * 0.75, vline_text_y, "Wolf-Rayet", rotation=90, fontsize=9, c="k")
    # LMXB 4U 1700+24
    D_LMXB = 0.42
    ax.vlines(D_LMXB, Y_MIN, Y_MAX, colors="k", linestyles="--")
    ax.text(
        D_LMXB * 1.1, vline_text_y, "LMXB 4U 1700+24", rotation=90, fontsize=9, c="k"
    )
    # MSP J0427-4715
    D_MSP = 0.16
    ax.vlines(D_MSP, Y_MIN, Y_MAX, colors="k", linestyles="--")
    ax.text(
        D_MSP * 0.75, vline_text_y, "MSP J0427-4715", rotation=90, fontsize=9, c="k"
    )

    ax.set_xlim([0.1, 10])
    ax.set_ylim(Y_MIN, Y_MAX)

    ax.set_yscale("log")
    ax.set_xscale("log")

    ax.set_ylabel(r"$N_{\mathrm{src}}$", fontsize=16)
    ax.set_xlabel(r"$d_{\mathrm{src}} \ [\mathrm{kpc}]$", fontsize=16)

    # ax.text(
    #     1.5,
    #     3e4,
    #     r"$\mathrm{GECCO (best-case)}$",
    #     rotation=-50,
    #     fontdict={"size": 12, "color": COLORS[0]},
    # )

    ax.legend(frameon=False, fontsize=10)


def make_plot_nearby_sources(ax):

    phis = [
        PHI_M31,
        PHI_M33,
        PHI_DRACO,
        PHI_URSA_MINOR,
        PHI_FORNAX_CL,
        PHI_COMA_CL,
    ]
    for i, phi in enumerate(phis):
        ax.errorbar(
            2 * i + 1.0,
            phi.nevents,
            phi.uncertainty,
            fmt="o",
            elinewidth=2,
            capsize=5,
            color=phi.plot_color,
        )

    idxs = 2 * np.arange(len(phis)) + 1
    ax.set_xticks(idxs)
    ax.set_xticklabels([phi[2] for phi in phis])
    plt.setp(ax2.get_xticklabels(), rotation=45)

    X_MIN = 0
    X_MAX = np.max(idxs) + 2
    ax.hlines(PHI_GECCO_BC, X_MIN, X_MAX, colors="k", label="GECCO (best-case)")
    ax.hlines(
        PHI_GECCO_WC,
        X_MIN,
        X_MAX,
        colors="k",
        label="GECCO (conservative)",
        linestyle="--",
    )

    ax.set_xlim([X_MIN, X_MAX])

    ax.set_yscale("log")

    ax.set_ylabel(r"$\phi_{511} \ [\mathrm{cm}^{-2} \ \mathrm{s}^{-1}]$", fontsize=16)

    ax.legend(frameon=False, fontsize=12)


if __name__ == "__main__":

    FIG_WIDTH = 7
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
    ax1 = axes[0]
    ax2 = axes[1]

    Y_MIN = 1e3
    Y_MAX = 4e7

    make_plot_point_src_vs_dist(ax1)
    make_plot_nearby_sources(ax2)

    plt.tight_layout()
    plt.savefig(THIS_DIR.joinpath("511.pdf"))
