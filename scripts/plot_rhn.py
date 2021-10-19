import h5py
import numpy as np
import warnings

# from matplotlib.lines import Line2D
# from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from hazma.rh_neutrino import RHNeutrino

from mpl_conf import COLOR_DICT as color_dict
from mpl_conf import LABEL_DICT as label_dict
from mpl_conf import SIGV_TEX, SIGV_UNITS, MEV_UNITS
import mpl_conf

Group = h5py.Group


DATAFILE_E: Group = h5py.File("../results/rhn_e.hdf5", "r")
DATAFILE_M: Group = h5py.File("../results/rhn_mu.hdf5", "r")

YLABEL = SIGV_TEX + r"$ \ $" + SIGV_UNITS
XLABEL = r"$m_{\chi}$" + r"$ \ $" + MEV_UNITS


def add_gecco(axis, masses, gecco):
    for key in gecco.keys():
        conf = {"color": color_dict[key]}
        c1 = 1 / gecco[key]["limits"][0, :]
        c2 = 1 / gecco[key]["limits"][1, :]
        avg = np.exp(np.log(c1 * c2) / 2.0)
        axis.fill_between(masses, c1, c2, lw=1, alpha=0.2, **conf)
        axis.plot(masses, avg, lw=2.5, **conf)
        axis.plot(masses, c1, lw=1.5, alpha=0.3, ls="-", **conf)
        axis.plot(masses, c2, lw=1.5, alpha=0.3, ls="-", **conf)


def existing_outline(datafile):
    egret = 1 / datafile["egret"][:]
    comptel = 1 / datafile["comptel"][:]
    fermi = 1 / datafile["fermi"][:]
    integral = 1 / datafile["integral"][:]
    return np.array(
        [np.max([e, c, f, i]) for e, c, f, i in zip(egret, comptel, fermi, integral)]
    )


def add_existing(axis, masses, ylims, datafile):
    existing = existing_outline(datafile)
    existing = np.clip(existing, np.min(ylims), np.max(ylims))
    axis.fill_between(masses, existing, 0.0, alpha=0.5)
    axis.plot(masses, existing)


def add_theta_contour(axis, datafile, theta, lepton):
    masses = datafile["masses"][:]
    rhn = RHNeutrino(1, theta, lepton)
    taus = np.zeros_like(masses)
    for i, m in enumerate(masses):
        rhn.mx = m
        taus[i] = 1 / (rhn.decay_widths()["total"] * 1.51927e21)
    axis.plot(masses, taus, ls="-.", color=mpl_conf.PURPLE, alpha=0.5, lw=1)


def add_plot(axis, datafile, ylims):
    masses = datafile["masses"][:]
    gecco = datafile["gecco"]

    add_existing(axis, masses, ylims, datafile)
    add_gecco(axis, masses, gecco)

    axis.set_yscale("log")
    axis.set_xscale("log")
    axis.set_ylim(ylims)
    axis.set_xlim(np.min(masses), np.max(masses))


def add_gecco_legend(axis, geccos):
    handels = []
    for key in geccos.keys():
        handels += [
            Line2D(
                [0],
                [0],
                color=color_dict[key],
                label="GECCO" + label_dict[key],
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        add_theta_contour(ax1, DATAFILE_E, 1e-18, "e")
        ax1.text(
            170,
            1e27,
            r"$\theta=10^{-18}$",
            fontsize=7,
            rotation=-70,
            color=mpl_conf.PURPLE,
        )
        add_theta_contour(ax1, DATAFILE_E, 1e-16, "e")
        ax1.text(
            40,
            1e27,
            r"$\theta=10^{-16}$",
            fontsize=7,
            rotation=-70,
            color=mpl_conf.PURPLE,
        )
        add_theta_contour(ax1, DATAFILE_E, 1e-14, "e")
        add_theta_contour(ax1, DATAFILE_E, 1e-12, "e")
        add_theta_contour(ax1, DATAFILE_E, 1e-10, "e")
        add_theta_contour(ax1, DATAFILE_E, 1e-8, "e")
        add_plot(ax1, DATAFILE_E, ylims)
        add_gecco_legend(ax1, DATAFILE_E["gecco"])

        add_theta_contour(ax2, DATAFILE_M, 1e-18, "mu")
        ax2.text(
            170,
            1e27,
            r"$\theta=10^{-18}$",
            fontsize=7,
            rotation=-70,
            color=mpl_conf.PURPLE,
        )
        add_theta_contour(ax2, DATAFILE_M, 1e-16, "mu")
        ax2.text(
            40,
            1e27,
            r"$\theta=10^{-16}$",
            fontsize=7,
            rotation=-70,
            color=mpl_conf.PURPLE,
        )
        add_theta_contour(ax2, DATAFILE_M, 1e-14, "mu")
        ax2.text(
            7,
            1e27,
            r"$\theta=10^{-14}$",
            fontsize=7,
            rotation=-70,
            color=mpl_conf.PURPLE,
        )
        add_theta_contour(ax2, DATAFILE_M, 1e-12, "mu")
        ax2.text(
            1,
            1e27,
            r"$\theta=10^{-12}$",
            fontsize=7,
            rotation=-70,
            color=mpl_conf.PURPLE,
        )
        add_theta_contour(ax2, DATAFILE_M, 1e-10, "mu")
        ax2.text(
            0.17,
            1e27,
            r"$\theta=10^{-10}$",
            fontsize=7,
            rotation=-70,
            color=mpl_conf.PURPLE,
        )
        add_theta_contour(ax2, DATAFILE_M, 1e-8, "mu")
        add_plot(ax2, DATAFILE_M, ylims)

    ax1.set_ylabel(YLABEL, fontdict={"size": 16})
    ax1.set_xlabel(XLABEL, fontdict={"size": 16})
    ax1.set_title(r"$\ell=e$", fontdict={"size": 16})
    ax1.grid(True, axis="y", which="major")
    ax1.grid(True, axis="x", which="major")
    ax1.set_yticks([10.0 ** x for x in range(21, 30)])

    ax2.set_xlabel(XLABEL, fontdict={"size": 16})
    ax2.set_title(r"$\ell=\mu$", fontdict={"size": 16})
    ax2.grid(True, axis="y", which="major")
    ax2.grid(True, axis="x", which="major")

    plt.tight_layout()
    plt.savefig("rhn.pdf")
