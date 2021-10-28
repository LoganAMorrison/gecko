import h5py
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from hazma.cmb import vx_cmb
from hazma.scalar_mediator import HiggsPortal

import utils
from utils import COLOR_DICT as color_dict
from utils import LABEL_DICT as label_dict
from utils import SIGV_TEX, SIGV_UNITS, MEV_UNITS
from utils import get_gecco_keys, strip_gecco

try:
    from gecko.utils import sigmav
except ImportError:
    import sys

    sys.path.append("..")
    from gecko.utils import sigmav

Group = h5py._hl.group.Group  # type: ignore


def add_cmb(axis, masses, datafile):
    cmb = datafile["cmb"][:]
    vratios = np.array([(1e-3 / vx_cmb(mx, 1e-4)) ** 2 for mx in masses])
    cmb_p_wave = vratios * cmb
    axis.plot(masses, cmb_p_wave, ls="--", c="k")


def add_rd(axis, masses, datafile):
    rd = datafile["relic-density"][:]
    axis.plot(masses, rd, ls="dotted", c="k", lw=1)


def add_pheno(axis, masses, datafile):
    pheno = datafile["pheno"][:]
    existing = utils.existing_outline(datafile)
    axis.fill_between(masses, pheno, existing, alpha=0.2, color=utils.ORANGE)
    axis.plot(masses, pheno, alpha=0.2, ls="--", color=utils.ORANGE)


def add_gsxx_contour(axis, datafile, ms_mx_ratio, gsxx, vx=1e-3, stheta=1e-3):
    masses = datafile["masses"][:]
    hp = HiggsPortal(mx=1, ms=1, gsxx=gsxx, stheta=stheta)
    svs = np.zeros_like(masses)
    for i, m in enumerate(masses):
        hp.mx = m
        hp.ms = m * ms_mx_ratio
        svs[i] = sigmav(hp, vx)
    axis.plot(masses, svs, ls="dotted", color=utils.RED, lw=1)


def add_gsxx_label(axis, x, y, gsxx, st=None, rotation=0):
    if st is not None:
        axis.text(
            x,
            y,
            r"$(g_{S\chi},s_{\theta}) = $(" + gsxx + "," + st + ")",
            fontsize=8,
            rotation=rotation,
            color=utils.RED,
        )
    else:
        axis.text(
            x,
            y,
            r"$g_{S\chi}$ = " + gsxx,
            fontsize=8,
            rotation=rotation,
            color=utils.RED,
        )


def add_plot(axis, datafile, ylims, include_pheno=False):
    masses = datafile["masses"][:]

    if include_pheno:
        add_pheno(axis, masses, datafile)

    add_cmb(axis, masses, datafile)
    add_rd(axis, masses, datafile)
    utils.add_existing_outline(axis, masses, ylims, datafile)
    utils.add_gecco(axis, masses, datafile)

    axis.set_yscale("log")
    axis.set_xscale("log")
    axis.set_ylim(ylims)
    axis.set_xlim(np.min(masses), np.max(masses))


def make_legend_axis(axis, datafile):
    axis.clear()
    axis.set_axis_off()
    handels = [
        Line2D([0], [0], color="k", label=label_dict["cmb"], ls="--", lw=1),
        Line2D([0], [0], color="k", label=label_dict["rd"], ls="dotted", lw=1),
    ]
    for key in get_gecco_keys(datafile):
        name = strip_gecco(key)
        handels += [
            Line2D(
                [0],
                [0],
                color=color_dict[name],
                label="GECCO" + label_dict[name],
                alpha=1,
            )
        ]
    handels += [
        Patch(color=color_dict["pheno"], label=label_dict["pheno"], alpha=0.3),
    ]
    handels += [
        Patch(color=color_dict["existing"], label=label_dict["existing"], alpha=0.3),
    ]
    axis.legend(handles=handels, loc="center", fontsize=10)


if __name__ == "__main__":
    fig = plt.figure(constrained_layout=True, dpi=150, figsize=(9, 3))
    widths = [2, 2, 1]
    spec = fig.add_gridspec(ncols=3, nrows=1, width_ratios=widths)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])
    ax3 = fig.add_subplot(spec[0, 2])

    ylims = (1e-37, 1e-23)

    DATAFILE_1_5: Group = h5py.File("results/higgs_portal_1_5.hdf5", "r")
    DATAFILE_0_5: Group = h5py.File("results/higgs_portal_0_5.hdf5", "r")

    YLABEL = SIGV_TEX + r"$ \ $" + SIGV_UNITS
    XLABEL = r"$m_{\chi}$" + r"$ \ $" + MEV_UNITS

    add_plot(ax1, DATAFILE_0_5, ylims)
    add_plot(ax2, DATAFILE_1_5, ylims, True)

    add_gsxx_contour(ax1, DATAFILE_0_5, 0.5, 1, vx=1e-3, stheta=1)
    add_gsxx_label(ax1, 100, 1e-25, r"$1$", rotation=-23)
    add_gsxx_contour(ax1, DATAFILE_0_5, 0.5, 1e-2, vx=1e-3, stheta=1)
    add_gsxx_label(ax1, 100, 2e-34, r"$10^{-2}$", rotation=-25)
    add_gsxx_contour(ax1, DATAFILE_0_5, 0.5, 1e-4, vx=1e-3, stheta=1)
    add_gsxx_label(ax1, 2, 5e-37, r"$10^{-4}$", rotation=-25)

    add_gsxx_contour(ax2, DATAFILE_1_5, 1.5, 4 * np.pi, vx=1e-3, stheta=1)
    add_gsxx_label(ax2, 5, 8e-33, r"$4\pi$", r"$1$")

    make_legend_axis(ax3, DATAFILE_0_5)

    ax1.set_ylabel(YLABEL, fontdict={"size": 16})
    ax1.set_xlabel(XLABEL, fontdict={"size": 16})
    ax1.set_title(r"$m_{S}=0.5m_{\chi}$", fontdict={"size": 16})
    ax2.set_xlabel(XLABEL, fontdict={"size": 16})
    ax2.set_title(r"$m_{S}=1.5m_{\chi}$", fontdict={"size": 16})

    for ax in (ax1, ax2):
        utils.configure_ticks(ax, minor_ticks="x")
        utils.add_xy_grid(ax)

    plt.savefig("figures/higgs_portal.pdf")
