import h5py
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from hazma.parameters import electron_mass as me

from utils import COLOR_DICT as color_dict
from utils import LABEL_DICT as label_dict
from utils import SIGV_TEX, SIGV_UNITS, MEV_UNITS
import utils

Group = h5py._hl.group.Group  # type: ignore


def add_gecco(masses, datafile):
    for key in utils.get_gecco_keys(datafile):
        conf = {"color": color_dict[utils.strip_gecco(key)]}
        c1 = datafile[key][:]
        c2 = 5 * datafile[key][:]
        avg = np.exp(np.log(c1 * c2) / 2.0)
        plt.fill_between(masses, c1, c2, lw=1, alpha=0.2, **conf)
        plt.plot(masses, avg, lw=2.5, **conf)
        plt.plot(masses, c1, lw=1.5, alpha=0.3, ls="-", **conf)
        plt.plot(masses, c2, lw=1.5, alpha=0.3, ls="-", **conf)


def add_cmb(axis, data):
    if data is not None:
        axis.plot(1e3 * data.T[0], data.T[1], lw=1, ls="--", c="k")


def make_legend_axis(datafile):
    handels = [
        Line2D([0], [0], color="k", label=label_dict["cmb"], ls="--", lw=1),
        Line2D([0], [0], color="k", label=label_dict["rd"], ls="dotted", lw=1),
    ]
    for key in utils.get_gecco_keys(datafile):
        name = utils.strip_gecco(key)
        handels += [
            Line2D(
                [0],
                [0],
                color=color_dict[name],
                label="GECCO" + label_dict[name],
                alpha=1.0,
            )
        ]
    handels += [
        Patch(color=color_dict["pheno"], label=label_dict["pheno"], alpha=0.3),
    ]
    handels += [
        Patch(color=color_dict["existing"], label=label_dict["existing"], alpha=0.3),
    ]
    plt.legend(handles=handels, bbox_to_anchor=(1.0, 0.75), fontsize=12)


if __name__ == "__main__":
    fig, axis = plt.subplots(dpi=150, figsize=(8, 4.5))

    DATAFILE: Group = h5py.File("results/kinetic_mixing.hdf5", "r")
    YLABEL = SIGV_TEX + r"$ \ $" + SIGV_UNITS
    XLABEL = r"$m_{\chi}$" + r"$ \ $" + MEV_UNITS

    masses = DATAFILE["masses"][:]
    egret = DATAFILE["egret"][:]
    comptel = DATAFILE["comptel"][:]
    fermi = DATAFILE["fermi"][:]
    integral = DATAFILE["integral"][:]
    cmb = DATAFILE["cmb"][:]
    pheno = DATAFILE["pheno"][:]
    rd = DATAFILE["relic-density"][:]

    ymax = 1e-23
    ymin = 1e-33

    existing = np.array(
        [np.min([e, c, f, i]) for e, c, f, i in zip(egret, comptel, fermi, integral)]
    )
    existing = np.clip(existing, ymin, ymax)
    axis.fill_between(masses, existing, ymax, alpha=0.2, color=utils.BLUE)
    axis.plot(masses, existing, color=utils.BLUE)

    axis.fill_between(masses, pheno, existing, alpha=0.2, color=utils.ORANGE)
    axis.plot(masses, pheno, alpha=0.2, ls="--", color=utils.ORANGE)

    # add_gecco(masses, DATAFILE)
    utils.add_gecco(axis, masses, DATAFILE)

    axis.plot(masses, cmb, ls="--", c="k")
    axis.plot(masses, rd, ls="dotted", c="k", lw=1)

    make_legend_axis(DATAFILE)

    axis.set_yscale("log")
    axis.set_xscale("log")
    axis.set_title(r"$m_{\chi} = m_{V}/3$", fontdict={"size": 16})
    axis.set_ylim([ymin, ymax])
    axis.set_xlim([me * 1.12, np.max(masses)])

    axis.set_ylabel(YLABEL, fontdict={"size": 16})
    axis.set_xlabel(XLABEL, fontdict={"size": 16})

    axis.set_yticks([10.0 ** x for x in range(-33, -22)])

    utils.configure_ticks(axis)
    utils.add_xy_grid(axis)

    plt.tight_layout()
    plt.savefig("figures/kinetic_mixing.pdf")
