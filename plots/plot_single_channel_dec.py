import h5py
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt


from hazma.parameters import muon_mass as mmu

import utils
from utils import COLOR_DICT as color_dict
from utils import LABEL_DICT as label_dict
from utils import EXISTINGS
from utils import DM_MASS_TEX

Group = h5py._hl.group.Group  # type: ignore


cmb_epem = np.genfromtxt("data/cmb_epem.csv", delimiter=",")
cmb_gg = np.genfromtxt("data/cmb_gamma_gamma.csv", delimiter=",")


def get_gecco_keys(datafile):
    return list(filter(lambda k: "gecco" in k, list(datafile.keys())))


def strip_gecco(key: str):
    return key.replace("gecco-", "")


def add_gecco(axis, masses, datafile):
    for key in get_gecco_keys(datafile):
        conf = {"color": color_dict[strip_gecco(key)]}
        c1 = 1 / datafile[key][:]
        c2 = 1 / (5 * datafile[key][:])
        avg = np.exp(np.log(c1 * c2) / 2.0)
        axis.fill_between(masses, c1, c2, lw=1, alpha=0.2, **conf)
        axis.plot(masses, avg, lw=2.5, **conf)
        axis.plot(masses, c1, lw=1.5, alpha=0.3, ls="-", **conf)
        axis.plot(masses, c2, lw=1.5, alpha=0.3, ls="-", **conf)


def add_existing(axis, masses, data):
    for tel in EXISTINGS:
        conf = {"color": color_dict[tel], "alpha": 0.2, "lw": 1.0}
        y = 1 / data[tel][:]
        axis.fill_between(masses, y, 0.0, where=y > 1e20, **conf)
        axis.plot(masses, y, **conf)


def add_cmb(axis, data):
    if data is not None:
        axis.plot(1e3 * data.T[0], data.T[1], lw=1, ls="--", c="k")


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
    axis.clear()
    axis.set_axis_off()
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

    DATAFILE = h5py.File("results/single_channel_dec.hdf5", "r")
    YLABEL = r"$\mathrm{DM} \ \mathrm{lifetime} \ \tau \ [\mathrm{s}]$"
    XLABEL = DM_MASS_TEX

    axes = [ax1, ax2, ax3]
    ylabels = [YLABEL, None, YLABEL]
    xlabels = [XLABEL] * 3

    ylims = [[1e22, 1e27], [1e25, 1e31], [1e22, 1e26]]
    xlims = [[1, 1e4], [0.1, 12], [mmu * 2 + 10, 1e4]]
    titles = [r"$e^{+}e^{-}$", r"$\gamma\gamma$", r"$\mu^{+}\mu^{-}$"]
    cmbs = [cmb_epem, cmb_gg, None]

    for i, (channel, axis) in enumerate(zip(DATAFILE.keys(), axes)):
        group: Group = DATAFILE[channel]
        masses = group["masses"][:]
        utils.add_gecco(axis, masses, group, decay=True)
        add_existing(axis, masses, group)
        add_cmb(axis, cmbs[i])
        configure_axis(
            xlabels[i],
            ylabels[i],
            xlims[i],
            ylims[i],
            "log",
            "log",
            titles[i],
        )

    make_legend_axis(last_axis, DATAFILE["e e"], ["cmb"])
    plt.tight_layout()
    plt.savefig("figures/single_channel_dec.pdf")
