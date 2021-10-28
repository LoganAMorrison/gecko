import h5py
import numpy as np
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from hazma.rh_neutrino import RHNeutrino

import utils
from utils import COLOR_DICT as color_dict
from utils import LABEL_DICT as label_dict
from utils import get_gecco_keys, strip_gecco

Group = h5py.Group


def add_theta_contour(axis, datafile, theta, lepton):
    masses = datafile["masses"][:]
    rhn = RHNeutrino(1, theta, lepton)
    taus = np.zeros_like(masses)
    for i, m in enumerate(masses):
        rhn.mx = m
        taus[i] = 1 / (rhn.decay_widths()["total"] * 1.51927e21)
    axis.plot(masses, taus, ls="-.", color=utils.PURPLE, alpha=0.5, lw=1)


def add_theta_label(axis, x, y, p):
    axis.text(
        x,
        y,
        r"$\theta=10^{" + str(p) + r"}$",
        fontsize=7,
        rotation=-70,
        color=utils.PURPLE,
    )


def add_plot(axis, datafile, ylims):
    masses = datafile["masses"][:]

    utils.add_existing_outline(axis, masses, ylims, datafile, decay=True)
    utils.add_gecco(axis, masses, datafile, decay=True)

    axis.set_yscale("log")
    axis.set_xscale("log")
    axis.set_ylim(ylims)
    axis.set_xlim(np.min(masses), np.max(masses))


def add_gecco_legend(axis, datafile):
    handels = []
    for key in get_gecco_keys(datafile):
        name = strip_gecco(key)
        handels += [
            Line2D(
                [0],
                [0],
                color=color_dict[name],
                label="GECCO" + label_dict[name],
                alpha=0.7,
            )
        ]
    handels += [
        Patch(color=color_dict["existing"], label=r"$\gamma$-ray telescopes", alpha=0.5)
    ]
    axis.legend(handles=handels, loc=2, fontsize=10, frameon=False)


if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, dpi=150, figsize=(8, 4))
    ylims = (1e21, 1e29)

    DATAFILE_E: Group = h5py.File("results/rhn_e.hdf5", "r")
    DATAFILE_M: Group = h5py.File("results/rhn_mu.hdf5", "r")

    YLABEL = utils.TAU_TEX
    XLABEL = utils.RHN_MASS_TEX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        add_gecco_legend(ax1, DATAFILE_E)

        text_xs = [170, 40, 7, 1, 0.17]
        text_ys = [1e27, 1e27, 1e27, 1e27, 1e27]
        pows = [-18, -16, -14, -12, -10]

        for x, y, p in zip(text_xs, text_ys, pows):
            add_theta_contour(ax1, DATAFILE_E, 10 ** p, "e")
            if p < -15:
                add_theta_label(ax1, x, y, p)
            add_theta_contour(ax2, DATAFILE_M, 10 ** p, "mu")
            add_theta_label(ax2, x, y, p)

        add_plot(ax1, DATAFILE_E, ylims)
        add_plot(ax2, DATAFILE_M, ylims)

    ax1.set_ylabel(YLABEL, fontdict={"size": 16})
    ax1.set_xlabel(XLABEL, fontdict={"size": 16})
    ax1.set_title(r"$\ell=e$", fontdict={"size": 16})
    ax1.set_yticks([10.0 ** x for x in range(21, 30)])

    ax2.set_xlabel(XLABEL, fontdict={"size": 16})
    ax2.set_title(r"$\ell=\mu$", fontdict={"size": 16})
    ax2.set_yticks([10.0 ** x for x in range(21, 30)])

    for ax in (ax1, ax2):
        utils.configure_ticks(ax)
        utils.add_xy_grid(ax)

    plt.tight_layout()
    plt.savefig("figures/rhn.pdf")

    DATAFILE_E.close()
    DATAFILE_M.close()
