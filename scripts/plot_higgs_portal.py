import h5py
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from hazma.cmb import vx_cmb
from hazma.scalar_mediator import HiggsPortal

from mpl_conf import COLOR_DICT as color_dict
from mpl_conf import LABEL_DICT as label_dict
from mpl_conf import SIGV_TEX, SIGV_UNITS, MEV_UNITS
import mpl_conf

try:
    from gecko.utils import sigmav
except ImportError:
    import sys

    sys.path.append("..")
    from gecko.utils import sigmav

Group = h5py._hl.group.Group

DATAFILE_1_5: Group = h5py.File("../results/higgs_portal_1_5.hdf5", "r")
DATAFILE_0_5: Group = h5py.File("../results/higgs_portal_0_5.hdf5", "r")

YLABEL = SIGV_TEX + r"$ \ $" + SIGV_UNITS
XLABEL = r"$m_{\chi}$" + r"$ \ $" + MEV_UNITS


def add_gecco(axis, masses, datafile):
    gecco = datafile["gecco"]
    for key in gecco.keys():
        conf = {"color": color_dict[key]}
        c1 = gecco[key]["limits"][0, :]
        c2 = gecco[key]["limits"][1, :]
        avg = np.exp(np.log(c1 * c2) / 2.0)
        axis.fill_between(masses, c1, c2, lw=1, alpha=0.2, **conf)
        axis.plot(masses, avg, lw=2.5, **conf)
        axis.plot(masses, c1, lw=1.5, alpha=0.3, ls="-", **conf)
        axis.plot(masses, c2, lw=1.5, alpha=0.3, ls="-", **conf)


def add_cmb(axis, masses, datafile):
    cmb = datafile["cmb"][:]
    vratios = np.array([(1e-3 / vx_cmb(mx, 1e-4)) ** 2 for mx in masses])
    cmb_p_wave = vratios * cmb
    axis.plot(masses, cmb_p_wave, ls="--", c="k")


def add_rd(axis, masses, datafile):
    rd = datafile["relic-density"][:]
    axis.plot(masses, rd, ls="dotted", c="k", lw=1)


def existing_outline(datafile):
    egret = datafile["egret"][:]
    comptel = datafile["comptel"][:]
    fermi = datafile["fermi"][:]
    integral = datafile["integral"][:]
    return np.array(
        [np.min([e, c, f, i]) for e, c, f, i in zip(egret, comptel, fermi, integral)]
    )


def add_existing(axis, masses, ylims, datafile):
    existing = existing_outline(datafile)
    existing = np.clip(existing, np.min(ylims), np.max(ylims))
    axis.fill_between(masses, existing, np.max(ylims), alpha=0.5)
    axis.plot(masses, existing)


def add_pheno(axis, masses, datafile):
    pheno = datafile["pheno"][:]
    existing = existing_outline(datafile)
    axis.fill_between(masses, pheno, existing, alpha=0.2, color=mpl_conf.ORANGE)
    axis.plot(masses, pheno, alpha=0.2, ls="--", color=mpl_conf.ORANGE)


def add_gsxx_contour(axis, datafile, ms_mx_ratio, gsxx, vx=1e-3, stheta=1e-3):
    masses = datafile["masses"][:]
    hp = HiggsPortal(mx=1, ms=1, gsxx=gsxx, stheta=stheta)
    svs = np.zeros_like(masses)
    for i, m in enumerate(masses):
        hp.mx = m
        hp.ms = m * ms_mx_ratio
        svs[i] = sigmav(hp, vx)
    axis.plot(masses, svs, ls="dotted", color=mpl_conf.RED)


def add_plot(axis, datafile, ylims, include_pheno=False):
    masses = datafile["masses"][:]

    if include_pheno:
        add_pheno(axis, masses, datafile)

    add_cmb(axis, masses, datafile)
    add_rd(axis, masses, datafile)
    add_existing(axis, masses, ylims, datafile)
    add_gecco(axis, masses, datafile)

    axis.set_yscale("log")
    axis.set_xscale("log")
    axis.set_ylim(ylims)
    axis.set_xlim(np.min(masses), np.max(masses))


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
    plt.legend(handles=handels, loc="center", fontsize=12)


if __name__ == "__main__":
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, dpi=150, figsize=(9, 3))
    ylims = (1e-40, 1e-23)
    add_plot(ax1, DATAFILE_0_5, ylims)
    add_plot(ax2, DATAFILE_1_5, ylims, True)

    add_gsxx_contour(ax1, DATAFILE_0_5, 0.5, 1, vx=1e-3, stheta=1)
    add_gsxx_contour(ax1, DATAFILE_0_5, 0.5, 1e-2, vx=1e-3, stheta=1)
    add_gsxx_contour(ax1, DATAFILE_0_5, 0.5, 1e-4, vx=1e-3, stheta=1)

    add_gsxx_contour(ax2, DATAFILE_1_5, 1.5, 4 * np.pi, vx=1e-3, stheta=1)

    gecco = DATAFILE_0_5["gecco"]

    ax3.clear()
    ax3.set_axis_off()
    handels = [
        Line2D([0], [0], color="k", label=label_dict["cmb"], ls="--", lw=1),
        Line2D([0], [0], color="k", label=label_dict["rd"], ls="dotted", lw=1),
        Patch(color=color_dict["existing"], label=label_dict["existing"], alpha=0.3),
        Patch(color=color_dict["pheno"], label=label_dict["pheno"], alpha=0.3),
    ]
    for key in gecco.keys():
        handels += [
            Patch(color=color_dict[key], label="GECCO" + label_dict[key], alpha=0.7)
        ]
    ax3.legend(handles=handels, loc="center", fontsize=12)

    ax1.set_ylabel(YLABEL, fontdict={"size": 16})
    ax1.set_xlabel(XLABEL, fontdict={"size": 16})
    ax1.set_title(r"$m_{S}=0.5m_{\chi}$", fontdict={"size": 16})
    ax2.set_xlabel(XLABEL, fontdict={"size": 16})
    ax2.set_title(r"$m_{S}=1.5m_{\chi}$", fontdict={"size": 16})

    plt.tight_layout()
    plt.savefig("higgs_portal.pdf")
