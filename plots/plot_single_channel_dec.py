import h5py
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt


from hazma.parameters import muon_mass as mmu

from mpl_conf import COLOR_DICT as color_dict
from mpl_conf import LABEL_DICT as label_dict
from mpl_conf import EXISTINGS
from mpl_conf import GECCO_DECS
from mpl_conf import BR_TEX, SIGV_TEX, SIGV_UNITS, DM_MASS_TEX

Group = h5py._hl.group.Group  # type: ignore


cmb_epem = np.genfromtxt("data/cmb_epem.csv", delimiter=",")
cmb_gg = np.genfromtxt("data/cmb_gamma_gamma.csv", delimiter=",")


def add_gecco(axis, masses, data_low, data_high):
    for key in data_low.keys():
        conf = {"color": color_dict[key]}
        low = 1.0 / data_high[key][:]
        high = 1.0 / data_low[key][:]
        avg = np.exp(np.log(high * low) / 2.0)
        axis.fill_between(masses, high, low, lw=1, alpha=0.2, **conf)
        axis.plot(masses, avg, lw=2.5, **conf)
        axis.plot(masses, high, lw=1.5, alpha=0.3, ls="-", **conf)
        axis.plot(masses, low, lw=1.5, alpha=0.3, ls="-", **conf)


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


def make_legend_axis(axis, geccos, existings, cmbs):
    axis.clear()
    axis.set_axis_off()
    handels = []
    for key in geccos:
        handels += [
            Patch(color=color_dict[key], label="GECCO" + label_dict[key], alpha=0.7)
        ]
    for cmb in cmbs:
        handels += [Line2D([0], [0], color="k", label=label_dict[cmb], ls="--", lw=1)]
    for key in existings:
        handels += [Patch(color=color_dict[key], label=label_dict[key], alpha=0.3)]
    axis.legend(handles=handels, loc="center", fontsize=12)


if __name__ == "__main__":
    fig, ((ax1, ax2), (ax3, last_axis)) = plt.subplots(
        nrows=2, ncols=2, figsize=(8, 6.5)
    )
    fig.dpi = 150

    DATAFILE = h5py.File("results/single_channel_dec.hdf5", "r")
    YLABEL = BR_TEX + r"\times" + SIGV_TEX + r"$ \ $" + SIGV_UNITS
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
        gecco5 = group["gecco-5sigma"]
        gecco25 = group["gecco-25sigma"]
        add_gecco(axis, masses, gecco25, gecco5)
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

    make_legend_axis(last_axis, GECCO_DECS, EXISTINGS, ["cmb"])
    plt.tight_layout()
    plt.savefig("figures/single_channel_dec.pdf")
