import os
from pathlib import Path
from typing import Optional, Dict
import json
import argparse

import numpy as np
from rich.progress import Progress
import h5py

from hazma.parameters import (
    electron_mass as me,
    muon_mass as mmu,
    charged_pion_mass as mpi,
)


from .higgs_portal import HiggsPortalConstrainer
from .kinetic_mixing import KineticMixingConstrainer
from .rh_neutrino import RHNeutrinoConstrainer
from .single_channel import SingleChannelConstrainer
from .pbh import PbhConstrainer


this_dir = os.path.dirname(os.path.abspath(__file__))
head, _ = os.path.split(this_dir)
default_prefix = os.path.join(head, "results")

parser = argparse.ArgumentParser(
    prog="generate",
    description="Compute constraints on the Higgs-portal model.",
)
parser.add_argument("config", help="path to json file containing config.")

"""
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
    conf["mx-min"] = config.get("mx-min", 0.1)
    conf["mx-max"] = config.get("mx-max", 250.0)
    conf["prefix"] = config.get("prefix", default_prefix)
    conf["sigma"] = config.get("sigma", [5.0, 25.0])
    conf["overwrite"] = config.get("overwrite", False)

    return conf
"""


def parse_json(args, requires):
    with open(args.config, "r") as f:
        config = json.load(f)
    conf = {}

    required = ["filename"]
    required.append(requires)

    for key in required:
        if key not in config:
            raise ValueError(f"Config is required to specify {key}.")
        else:
            conf[key] = config[key]

    # These are optional
    conf["mx-min"] = config.get("mx-min", 0.1)
    conf["mx-max"] = config.get("mx-max", 250.0)
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


def simplified_model(
    model, prefix, filename, mx_min, mx_max, mass_ratio, sigmas, overwrite
):
    if model == "kinetic_mixing":
        simp = KineticMixingConstrainer()
    else:
        simp = HiggsPortalConstrainer()

    mxs = np.geomspace(mx_min, mx_max, 100)
    with Progress(transient=True) as progress:
        constraints = simp.constrain(mxs, mass_ratio, progress=progress, gecco=False)
        constraints["masses"] = mxs
        constraints["gecco"] = {
            key: {
                "sigma": [],
                "limits": [],
            }
            for key in simp._gecco_targets.keys()
        }
        for sigma in sigmas:
            geccos_ = simp.constrain(
                mxs,
                ms_mx_ratio=mass_ratio,
                sigma=sigma,
                progress=progress,
                cmb=False,
                existing=False,
                pheno=False,
                relic_density=False,
            )
            for target, svs in geccos_.items():
                constraints["gecco"][target]["sigma"].append(sigma)
                constraints["gecco"][target]["limits"].append(svs)

    write_data(prefix, filename, constraints, overwrite)


def rh_neutrino(prefix, filename, mx_min, mx_max, lepton, sigmas, overwrite):
    rhn = RHNeutrinoConstrainer()
    mxs = np.geomspace(mx_min, mx_max, 100)
    with Progress(transient=True) as progress:
        constraints = rhn.constrain(mxs, lepton=lepton, progress=progress, gecco=False)
        constraints["masses"] = mxs
        constraints["gecco"] = {
            key: {
                "sigma": [],
                "limits": [],
            }
            for key in rhn._gecco_targets.keys()
        }
        for sigma in sigmas:
            geccos_ = rhn.constrain(
                mxs,
                sigma=sigma,
                lepton=lepton,
                progress=progress,
                existing=False,
            )
            for target, svs in geccos_.items():
                constraints["gecco"][target]["sigma"].append(sigma)
                constraints["gecco"][target]["limits"].append(svs)

    write_data(prefix, filename, constraints, overwrite)


def single_channel(decay, prefix, filename, sigmas, overwrite):
    singlechannel = SingleChannelConstrainer(decay)
    masses = {
        "e e": np.geomspace(me, 1e4, 100),
        "mu mu": np.geomspace(mmu, 1e4, 100),
        "g g": np.geomspace(1e-1, 1e4, 100),
        "pi pi": np.geomspace(mpi, 1e4, 100),
    }
    cmb = not decay
    constraints = {}
    with Progress(transient=True) as progress:
        for fs, mxs in masses.items():
            constraints[fs] = singlechannel.constrain(
                mxs, fs, decay=decay, progress=progress, gecco=False, cmb=cmb
            )
            constraints[fs]["masses"] = mxs
            targets = single_channel._gecco_targets
            constraints[fs]["gecco"] = {
                key: {
                    "sigma": [],
                    "limits": [],
                }
                for key in targets.keys()
            }
            for sigma in sigmas:
                geccos_ = singlechannel.constrain(
                    mxs,
                    fs,
                    decay=decay,
                    sigma=sigma,
                    progress=progress,
                    cmb=False,
                    existing=False,
                )
                for target, svs in geccos_.items():
                    constraints[fs]["gecco"][target]["sigma"].append(sigma)
                    constraints[fs]["gecco"][target]["limits"].append(svs)

    write_data(prefix, filename, constraints, overwrite)


def pbh(prefix, filename, sigmas, overwrite):
    pbh = PbhConstrainer()
    with Progress(transient=True) as progress:
        constraints = pbh.constrain(progress=progress, gecco=False, include_masses=True)
        constraints["gecco"] = {
            key: {
                "sigma": [],
                "limits": [],
            }
            for key in pbh._gecco_targets.keys()
        }
        for sigma in sigmas:
            geccos_ = pbh.constrain(
                progress=progress,
                existing=False,
            )
            for target, svs in geccos_.items():
                constraints["gecco"][target]["sigma"].append(sigma)
                constraints["gecco"][target]["limits"].append(svs)

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
        simplified_model(
            model,
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
            model == "single-channel-dec",
            config["prefix"],
            config["filename"],
            config["sigma"],
            config["overwrite"],
        )
    else:
        raise ValueError(f"Unknown model {model}.")
