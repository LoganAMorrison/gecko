import sys
import os

import h5py
import numpy as np
from rich.progress import Progress
import argparse

try:
    from gecko import HiggsPortalConstraints
    from gecko.utils import add_to_dataset
except ImportError:
    sys.path.append("..")
    from gecko import HiggsPortalConstraints
    from gecko.utils import add_to_dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def generate(filename, mass_ratio, sigmas, overwrite):
    hp = HiggsPortalConstraints()
    mxs = np.geomspace(0.1, 250.0, 100)
    hp_constraints = {}
    with Progress(transient=True) as progress:
        hp_constraints = hp.compute(mxs, mass_ratio, progress=progress, gecco=False)
        for sigma in sigmas:
            name = f"gecco-{sigma}sigma"
            hp_constraints[name] = hp.compute(
                mxs,
                ms_mx_ratio=mass_ratio,
                sigma=sigma,
                progress=progress,
                cmb=False,
                existing=False,
                pheno=False,
                relic_density=False,
            )

    fname = filename + ".hdf5"
    if os.path.exists(filename + ".hdf5"):
        if not overwrite:
            i = 1
            while os.path.exists(fname):
                fname = filename + "-" + str(i) + ".hdf5"
    hp_constraints["masses"] = mxs
    with h5py.File(fname, "w") as dataset:
        add_to_dataset(dataset, hp_constraints)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute constraints on the Higgs-portal dark matter model."
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
        "--mass-ratio",
        dest="mass_ratio",
        nargs="?",
        action="store",
        metavar="RATIO",
        type=float,
        default=0.5,
        help=" ".join(
            [
                "ratio of the scalar mediator mass to dark-matter mass.",
                "Default is 0.5.",
            ]
        ),
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
        metavar="FNAME",
        type=str,
        default="higgs_portal",
        help=" ".join(
            [
                "Name of the datafile for annihilation constraints.",
                "Default is 'higgs_portal_{mass-ratio}'.",
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
    mass_ratio = args.mass_ratio
    overwrite = args.overwrite
    sigmas = args.sigmas

    generate(filename, mass_ratio, sigmas, overwrite)
