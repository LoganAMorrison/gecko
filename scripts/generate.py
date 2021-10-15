import numpy as np
from rich.progress import Progress
import argparse
import os
from pathlib import Path
from typing import Optional, Dict
import json

import h5py

from hazma.parameters import (
    electron_mass as me,
    muon_mass as mmu,
    charged_pion_mass as mpi,
)


try:
    from gecko import HiggsPortalConstraints
    from gecko import KineticMixingConstraints
    from gecko import RHNeutrinoConstraints
    from gecko import SingleChannelConstraints
except ImportError:
    import sys

    sys.path.append("..")
    from gecko import HiggsPortalConstraints
    from gecko import KineticMixingConstraints
    from gecko import RHNeutrinoConstraints
    from gecko import SingleChannelConstraints


this_dir = os.path.dirname(os.path.abspath(__file__))
head, _ = os.path.split(this_dir)
default_prefix = os.path.join(head, "results")


def parse_json(args):
    with open(args.config, "r") as f:
        config = json.load(f)
    conf = {}
    # These are required
    if "model" not in config:
        raise ValueError("Config must specify model.")
    else:
        conf["model"] = config["model"]

    required = ["filename"]
    if config["model"] in ["higgs-portal", "kinetic-mixing"]:
        required += ["mass-ratio"]

    if config["model"] == "rh-neutrino":
        required += ["lepton"]

    for key in required:
        if key not in config:
            raise ValueError(f"Config is required to specify {key}.")
        else:
            conf[key] = config[key]

    # These are optional
    conf["mx-min"] = 0.1
    conf["mx-max"] = 250.0
    conf["prefix"] = config.get("prefix", default_prefix)
    conf["sigma"] = config.get("sigma", [5.0, 25.0])
    conf["overwrite"] = config.get("overwrite", False)

    return conf


def add_to_dataset(dataset, dictionary: Dict, path: Optional[str] = None):
    """Add a nested dictionary to the hdf5 dataset."""
    for k, v in dictionary.items():
        if path is not None:
            newpath = "/".join([path, k])
        else:
            newpath = f"{k}"
        if isinstance(v, dict):
            add_to_dataset(dataset, v, newpath)
        else:
            # at the bottom
            dataset.create_dataset(newpath, data=v)


def write_data(prefix: str, filename: str, dataset: Dict, overwrite: bool):
    """Write the dataset to the file."""
    fname = Path(prefix, filename).with_suffix(".hdf5")
    if fname.exists and not overwrite:
        i = 1
        while fname.exists():
            fname = fname.with_name(filename + "-" + str(i))
    with h5py.File(fname, "w") as f:
        add_to_dataset(f, dataset)


def higgs_portal(prefix, filename, mx_min, mx_max, mass_ratio, sigmas, overwrite):
    hp = HiggsPortalConstraints()
    mxs = np.geomspace(mx_min, mx_max, 100)
    constraints = {"masses": mxs}
    with Progress(transient=True) as progress:
        constraints = hp.compute(mxs, mass_ratio, progress=progress, gecco=False)
        for sigma in sigmas:
            name = f"gecco-{sigma}sigma"
            constraints[name] = hp.compute(
                mxs,
                ms_mx_ratio=mass_ratio,
                sigma=sigma,
                progress=progress,
                cmb=False,
                existing=False,
                pheno=False,
                relic_density=False,
            )

    write_data(prefix, filename, constraints, overwrite)


def kinetic_mixing(prefix, filename, mx_min, mx_max, mass_ratio, sigmas, overwrite):
    km = KineticMixingConstraints()
    mxs = np.geomspace(mx_min, mx_max, 100)
    constraints = {"masses": mxs}
    with Progress(transient=True) as progress:
        constraints = km.compute(mxs, mass_ratio, progress=progress, gecco=False)
        for sigma in sigmas:
            name = f"gecco-{sigma}sigma"
            constraints[name] = km.compute(
                mxs,
                ms_mx_ratio=mass_ratio,
                sigma=sigma,
                progress=progress,
                cmb=False,
                existing=False,
                pheno=False,
                relic_density=False,
            )

    write_data(prefix, filename, constraints, overwrite)


def single_channel(decay, prefix, filename, sigmas, overwrite):
    singlechannel = SingleChannelConstraints()
    masses = {
        "e e": np.geomspace(me, 1e4, 100),
        "mu mu": np.geomspace(mmu, 1e4, 100),
        "g g": np.geomspace(1e-1, 1e4, 100),
        "pi pi": np.geomspace(mpi, 1e4, 100),
    }
    cmb = False if decay else True
    constraints = {}
    with Progress(transient=True) as progress:
        for fs, mxs in masses.items():
            constraints[fs] = {"masses": mxs}
            constraints[fs] = singlechannel.compute(
                mxs, fs, decay=decay, progress=progress, gecco=False, cmb=cmb
            )
            for sigma in sigmas:
                name = f"gecco-{sigma}sigma"
                constraints[fs][name] = singlechannel.compute(
                    mxs,
                    fs,
                    decay=decay,
                    sigma=sigma,
                    progress=progress,
                    cmb=False,
                    existing=False,
                )

    write_data(prefix, filename, constraints, overwrite)


def rh_neutrino(prefix, filename, mx_min, mx_max, lepton, sigmas, overwrite):
    rhn = RHNeutrinoConstraints()
    mxs = np.geomspace(mx_min, mx_max, 100)
    constraints = {"masses": mxs}
    with Progress(transient=True) as progress:
        constraints = rhn.compute(mxs, lepton=lepton, progress=progress, gecco=False)
        for sigma in sigmas:
            name = f"gecco-{sigma}sigma"
            constraints[name] = rhn.compute(
                mxs,
                sigma=sigma,
                lepton=lepton,
                progress=progress,
                existing=False,
            )

    write_data(prefix, filename, constraints, overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="Compute constraints on the specified model.",
    )
    parser.add_argument("config", help="path to json file containing config.")
    config = parse_json(parser.parse_args())
    model = config["model"]

    if model in ["higgs-portal", "kinetic-mixing"]:
        call = higgs_portal if model == "higgs-portal" else kinetic_mixing
        call(
            config["prefix"],
            config["filename"],
            config["mx-min"],
            config["mx-max"],
            config["mass-ratio"],
            config["sigma"],
            config["overwrite"],
        )
    elif model == "rh-neutrino":
        rh_neutrino(
            config["prefix"],
            config["filename"],
            config["mx-min"],
            config["mx-max"],
            config["lepton"],
            config["sigma"],
            config["overwrite"],
        )
    elif model in ["single-channel-ann", "single-channel-dec"]:
        single_channel(
            True if model == "single-channel-ann" else False,
            config["prefix"],
            config["filename"],
            config["sigma"],
            config["overwrite"],
        )
    else:
        raise ValueError(f"Unknown model {model}.")
