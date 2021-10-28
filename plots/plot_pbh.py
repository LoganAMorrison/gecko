import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from hazma.parameters import MeV_to_g

import utils
from utils import COLOR_DICT as color_dict
from utils import LABEL_DICT as label_dict
from utils import get_gecco_keys, strip_gecco

Group = h5py._hl.group.Group  # type: ignore


def add_existing(axis: plt.Axes, masses, datafile, ylims):
    egret = datafile["egret"][:]
    comptel = datafile["comptel"][:]
    fermi = datafile["fermi"][:]
    integral = datafile["integral"][:]
    existing = np.array(
        [np.min([e, c, f, i]) for e, c, f, i in zip(egret, comptel, fermi, integral)]
    )
    existing = np.clip(existing, ylims[0], ylims[1])
    axis.fill_between(masses, existing, ylims[1], alpha=0.2, color=utils.BLUE)
    axis.plot(masses, existing, color=utils.BLUE)


def make_legend_axis(axis: plt.Axes, datafile):
    handels = []
    for key in get_gecco_keys(datafile):
        name = strip_gecco(key)
        handels += [
            Line2D(
                [0],
                [0],
                color=color_dict[name],
                label=label_dict[name],
                alpha=1.0,
            )
        ]
    handels += [
        Patch(color=color_dict["existing"], label=label_dict["existing"], alpha=0.3),
    ]
    axis.legend(handles=handels, fontsize=10, frameon=False)


if __name__ == "__main__":
    FIGSIZE = (5, 4)
    fig, axis = plt.subplots(dpi=150, figsize=FIGSIZE)

    DATAFILE: Group = h5py.File("results/pbh.hdf5", "r")
    YLABEL = r"$f_{\mathrm{PBH}}$"
    XLABEL = r"$M_{\mathrm{PBH}} \ [\mathrm{g}]$"

    masses = DATAFILE["masses"][:] * MeV_to_g

    ylims = (1e-6, 2.0)
    xlims = (2e15, 2e18)

    utils.add_existing_outline(axis, masses, ylims, DATAFILE)
    utils.add_gecco(axis, masses, DATAFILE)

    plt.plot(masses, np.ones_like(masses), ls="--", lw=2, color="k")

    make_legend_axis(axis, DATAFILE)

    utils.configure_ticks(axis)

    axis.set_yscale("log")
    axis.set_xscale("log")
    axis.set_ylim(ylims)
    axis.set_xlim(xlims)

    axis.set_ylabel(YLABEL, fontdict={"size": 16})
    axis.set_xlabel(XLABEL, fontdict={"size": 16})

    plt.tight_layout()
    plt.savefig("figures/pbh.pdf")
