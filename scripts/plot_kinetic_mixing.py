import h5py
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from hazma.parameters import electron_mass as me

from mpl_conf import COLOR_DICT as color_dict
from mpl_conf import LABEL_DICT as label_dict
from mpl_conf import SIGV_TEX, SIGV_UNITS, MEV_UNITS
import mpl_conf

Group = h5py._hl.group.Group

DATAFILE: Group = h5py.File("../results/kinetic_mixing.hdf5", "r")

YLABEL = SIGV_TEX + r"$ \ $" + SIGV_UNITS
XLABEL = r"$m_{\chi}$" + r"$ \ $" + MEV_UNITS


def add_gecco(masses, data_low, data_high):
    for key in data_low.keys():
        conf = {"color": color_dict[key]}
        low = data_low[key][:]
        high = data_high[key][:]
        avg = np.exp(np.log(high * low) / 2.0)
        plt.fill_between(masses, high, low, lw=1, alpha=0.2, **conf)
        plt.plot(masses, avg, lw=2.5, **conf)
        plt.plot(masses, high, lw=1.5, alpha=0.3, ls="-", **conf)
        plt.plot(masses, low, lw=1.5, alpha=0.3, ls="-", **conf)


def add_cmb(axis, data):
    if data is not None:
        axis.plot(1e3 * data.T[0], data.T[1], lw=1, ls="--", c="k")


def make_legend_axis(geccos, existing, pheno, cmb):
    handels = []
    for key in geccos:
        handels += [
            Patch(color=color_dict[key], label="GECCO" + label_dict[key], alpha=0.7)
        ]
    if cmb is not None:
        handels += [Line2D([0], [0], color="k", label=label_dict[cmb], ls="--", lw=1)]
    if existing is not None:
        handels += [
            Patch(color=color_dict[existing], label=label_dict[existing], alpha=0.3)
        ]
    if pheno is not None:
        handels += [Patch(color=color_dict[pheno], label=label_dict[pheno], alpha=0.3)]
    plt.legend(handles=handels, bbox_to_anchor=(1.0, 1.0), fontsize=12)


if __name__ == "__main__":
    fig = plt.figure(dpi=150, figsize=(8, 4.5))

    masses = DATAFILE["masses"][:]
    gecco5 = DATAFILE["gecco-5sigma"]
    gecco25 = DATAFILE["gecco-25sigma"]
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
    plt.fill_between(masses, existing, ymax, alpha=0.2, color=mpl_conf.BLUE)
    plt.plot(masses, existing, color=mpl_conf.BLUE)

    plt.fill_between(masses, pheno, existing, alpha=0.2, color=mpl_conf.ORANGE)
    plt.plot(masses, pheno, alpha=0.2, ls="--", color=mpl_conf.ORANGE)

    add_gecco(masses, gecco25, gecco5)

    plt.plot(masses, cmb, ls="--", c="k")
    plt.plot(masses, rd, ls="dotted", c="k", lw=1)

    plt.yscale("log")
    plt.xscale("log")
    plt.ylim([ymin, ymax])
    plt.xlim([me * 1.12, np.max(masses)])

    plt.ylabel(YLABEL, fontdict={"size": 16})
    plt.xlabel(XLABEL, fontdict={"size": 16})

    handels = [
        Line2D([0], [0], color="k", label=label_dict["cmb"], ls="--", lw=1),
        Line2D([0], [0], color="k", label=label_dict["rd"], ls="dotted", lw=1),
        Patch(color=color_dict["existing"], label=label_dict["existing"], alpha=0.3),
        Patch(color=color_dict["pheno"], label=label_dict["pheno"], alpha=0.3),
    ]
    for key in gecco5.keys():
        handels += [
            Patch(color=color_dict[key], label="GECCO" + label_dict[key], alpha=0.7)
        ]
    plt.legend(handles=handels, bbox_to_anchor=(1.0, 0.75), fontsize=12)

    plt.tight_layout()
    plt.savefig("kinetic_mixing.pdf")
