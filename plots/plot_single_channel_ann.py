import h5py
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt


from hazma.cmb import vx_cmb

import utils
from utils import COLOR_DICT as color_dict
from utils import LABEL_DICT as label_dict
from utils import BR_TEX, SIGV_TEX, SIGV_UNITS, DM_MASS_TEX

Group = h5py._hl.group.Group  # type: ignore

DATAFILE = h5py.File("results/single_channel_ann.hdf5", "r")

GECCOS = [
    "draco_nfw_1_arcmin_cone",
    "gc_ein_1_arcmin_cone_optimistic",
    "gc_nfw_1_arcmin_cone",
    "m31_nfw_1_arcmin_cone",
]

EXISTINGS = ["integral", "comptel", "egret", "fermi"]
CMBS = ["cmb_p_wave", "cmb_s_wave"]


def get_gecco_keys(datafile):
    return list(filter(lambda k: "gecco" in k, list(datafile.keys())))


def strip_gecco(key: str):
    return key.replace("gecco-", "")


def add_gecco(axis, masses, datafile):
    for key in get_gecco_keys(datafile):
        conf = {"color": color_dict[strip_gecco(key)]}
        c1 = datafile[key][:]
        c2 = 5 * datafile[key][:]
        avg = np.exp(np.log(c1 * c2) / 2.0)
        axis.fill_between(masses, c1, c2, lw=1, alpha=0.2, **conf)
        axis.plot(masses, avg, lw=2.5, **conf)
        axis.plot(masses, c1, lw=1.5, alpha=0.3, ls="-", **conf)
        axis.plot(masses, c2, lw=1.5, alpha=0.3, ls="-", **conf)


def add_existing(axis, masses, data):
    for tel in EXISTINGS:
        conf = {"color": color_dict[tel], "alpha": 0.2, "lw": 1.0}
        y = data[tel][:]
        axis.fill_between(masses, y, 1e-20, where=y < 1e-20, **conf)
        axis.plot(masses, y, **conf)


def add_cmb(axis, masses, data):
    cmb = data["cmb"][:]
    vratios = np.array([(1e-3 / vx_cmb(mx, 1e-4)) ** 2 for mx in masses])
    axis.plot(masses[cmb < 1e-20], vratios * cmb[cmb < 1e-20], lw=1, ls="--", c="k")
    axis.plot(masses[cmb < 1e-20], cmb[cmb < 1e-20], lw=1, ls="-.", c="k")


def configure_axis(xlabel, ylabel, xlims, ylims, yscale, xscale, title):
    if ylabel is not None:
        axis.set_ylabel(ylabel, fontsize=12)
    if xlabel is not None:
        axis.set_xlabel(xlabel, fontsize=12)
    if yscale is not None:
        axis.set_yscale(yscale)
    if xscale is not None:
        axis.set_xscale(xscale)
    if ylims is not None:
        axis.set_ylim(ylims)
    if xlims is not None:
        axis.set_xlim(xlims)
    if title is not None:
        axis.set_title(title, fontsize=16)


def make_legend_axis(axis, datafile, cmbs):
    last_axis.clear()
    last_axis.set_axis_off()
    handels = []
    for cmb in cmbs:
        handels += [Line2D([0], [0], color="k", label=label_dict[cmb], ls="--", lw=1)]
    for key in get_gecco_keys(datafile):
        name = strip_gecco(key)
        handels += [
            Line2D(
                [0],
                [0],
                color=color_dict[name],
                label="GECCO" + label_dict[name],
            )
        ]
    for key in ["egret", "integral", "fermi", "comptel"]:
        handels += [Patch(color=color_dict[key], label=label_dict[key], alpha=0.3)]
    axis.legend(handles=handels, loc="center", fontsize=12)


if __name__ == "__main__":
    fig, ((ax1, ax2), (ax3, last_axis)) = plt.subplots(
        nrows=2, ncols=2, figsize=(8, 6.5)
    )
    fig.dpi = 150

    YLABEL = BR_TEX + r"$\times$" + SIGV_TEX + r"$ \ $" + SIGV_UNITS
    XLABEL = DM_MASS_TEX

    axes = [ax1, ax2, ax3]
    ylabels = [YLABEL, None, YLABEL]
    xlabels = [XLABEL] * 3

    ylims = [[5e-33, 1e-22], [1e-37, 1e-28], [1e-30, 1e-22]]
    xmaxs = [1e4, 9, 1e4]
    titles = [r"$e^{+}e^{-}$", r"$\gamma\gamma$", r"$\mu^{+}\mu^{-}$"]

    for i, (channel, axis) in enumerate(zip(DATAFILE.keys(), axes)):
        group: Group = DATAFILE[channel]
        masses = group["masses"][:]
        utils.add_gecco(axis, masses, group)
        add_existing(axis, masses, group)
        add_cmb(axis, masses, group)
        configure_axis(
            xlabels[i],
            ylabels[i],
            (np.min(masses), xmaxs[i]),
            ylims[i],
            "log",
            "log",
            titles[i],
        )

    make_legend_axis(last_axis, DATAFILE["e e"], CMBS)
    plt.tight_layout()
    plt.savefig("figures/single_channel_ann.pdf")
