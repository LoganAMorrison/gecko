import sys
import os

import h5py
import numpy as np
from rich.progress import Progress
import argparse
from hazma.parameters import (
    electron_mass as me,
    muon_mass as mmu,
    charged_pion_mass as mpi,
)

try:
    from gecko import SingleChannelConstraints
    from gecko.utils import add_to_dataset
except ImportError:
    sys.path.append("..")
    from gecko import SingleChannelConstraints
    from gecko.utils import add_to_dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def generate(filename, sigmas, overwrite):
    singlechannel = SingleChannelConstraints()
    masses = {
        "e e": np.geomspace(me, 1e4, 100),
        "mu mu": np.geomspace(mmu, 1e4, 100),
        "g g": np.geomspace(1e-1, 1e4, 100),
        "pi pi": np.geomspace(mpi, 1e4, 100),
    }
    constraints = {}
    with Progress(transient=True) as progress:
        for fs, mxs in masses.items():
            constraints[fs] = singlechannel.compute(
                mxs, fs, decay=False, progress=progress, gecco=False
            )
            for sigma in sigmas:
                name = f"gecco-{sigma}sigma"
                constraints[fs][name] = singlechannel.compute(
                    mxs,
                    fs,
                    decay=False,
                    sigma=sigma,
                    progress=progress,
                    cmb=False,
                    existing=False,
                )

    fname = filename + ".hdf5"
    if os.path.exists(filename + ".hdf5"):
        if not overwrite:
            i = 1
            while os.path.exists(fname):
                fname = filename + "-" + str(i) + ".hdf5"

    with h5py.File(fname, "w") as dataset:
        add_to_dataset(dataset, constraints)
        add_to_dataset(dataset, {fs: {"masses": mxs} for fs, mxs in masses.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute constraints on single-channel dark matter models."
    )
    head, _ = os.path.split(THIS_DIR)
    default_prefix = os.path.join(head, "results")
    parser.add_argument(
        "--sigma",
        dest="sigmas",
        nargs="+",
        action="store",
        metavar="SIGMA",
        type=int,
        default=[5, 25],
        help=" ".join(["Sigma value(s) for discovery limit.", "Default is 5 25."]),
    )
    parser.add_argument(
        "--prefix",
        dest="prefix",
        nargs="?",
        action="store",
        metavar="PREFIX",
        type=str,
        default=default_prefix,
        help=" ".join(
            [
                "absolute path to directory where datafile will be stored.",
                f"Default is {default_prefix}",
            ]
        ),
    )
    parser.add_argument(
        "--filename",
        dest="filename",
        nargs="?",
        action="store",
        metavar="FILENAME",
        type=str,
        default="single_channel_ann",
        help=" ".join(
            [
                "Name of the datafile for annihilation constraints.",
                "Default is 'singal_channel_ann'.",
            ]
        ),
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_const",
        const=True,
        default=False,
        help=" ".join(
            [
                "If this flag is given, the file will be over-written",
                "if it exists.",
            ]
        ),
    )

    args = parser.parse_args()
    filename = os.path.join(args.prefix, args.filename)
    sigmas = args.sigmas
    overwrite = args.overwrite

    generate(filename, sigmas, overwrite)
