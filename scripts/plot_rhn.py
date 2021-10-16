import h5py
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from hazma.rh_neutrino import RHNeutrino
from hazma.parameters import sv_inv_MeV_to_cm3_per_s

from mpl_conf import COLOR_DICT as color_dict
from mpl_conf import LABEL_DICT as label_dict
from mpl_conf import SIGV_TEX, SIGV_UNITS, MEV_UNITS
import mpl_conf

Group = h5py._hl.group.Group

DATAFILE_E: Group = h5py.File("../results/rhn_e.hdf5", "r")
DATAFILE_M: Group = h5py.File("../results/rhn_mu.hdf5", "r")

YLABEL = SIGV_TEX + r"$ \ $" + SIGV_UNITS
XLABEL = r"$m_{\chi}$" + r"$ \ $" + MEV_UNITS


def add_gecco(axis, masses, data_low, data_high):
    for key in data_low.keys():
        conf = {"color": color_dict[key]}
        low = 1 / data_low[key][:]
        high = 1 / data_high[key][:]
        avg = np.exp(np.log(high * low) / 2.0)
        axis.fill_between(masses, high, low, lw=1, alpha=0.2, **conf)
        axis.plot(masses, avg, lw=2.5, **conf)
        axis.plot(masses, high, lw=1.5, alpha=0.3, ls="-", **conf)
        axis.plot(masses, low, lw=1.5, alpha=0.3, ls="-", **conf)


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


def add_theta_contour(axis, datafile, theta, lepton):
    masses = datafile["masses"][:]
    rhn = RHNeutrino(1, theta, lepton)
    taus = np.zeros_like(masses)
    for i, m in enumerate(masses):
        rhn.mx = m
        taus[i] = rhn.decay_widths()["total"]
    axis.plot(masses, taus, ls="dotted", color=mpl_conf.RED, alpha=0.2)


def add_plot(axis, datafile, ylims):
    masses = datafile["masses"][:]
    gecco5 = datafile["gecco-5sigma"]
    gecco25 = datafile["gecco-25sigma"]

    add_existing(axis, masses, ylims, datafile)
    add_gecco(axis, masses, gecco25, gecco5)

    axis.set_yscale("log")
    axis.set_xscale("log")
    axis.set_ylim(ylims)
    axis.set_xlim(np.min(masses), np.max(masses))


if __name__ == "__main__":
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, dpi=150, figsize=(8, 3))
    ylims = (1e-40, 1e-23)
    add_plot(ax1, DATAFILE_E, ylims)
    add_plot(ax2, DATAFILE_M, ylims)

    ax1.set_ylabel(YLABEL, fontdict={"size": 16})
    ax1.set_xlabel(XLABEL, fontdict={"size": 16})
    ax1.set_title(r"$\ell=e$", fontdict={"size": 16})
    ax2.set_xlabel(XLABEL, fontdict={"size": 16})
    ax2.set_title(r"$\ell=\mu$", fontdict={"size": 16})

    plt.tight_layout()
    plt.savefig("rhn.pdf")
