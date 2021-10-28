import os
from pathlib import Path
import json
import argparse
from typing import Union, Dict, Optional

import h5py
from rich.console import Console
from hazma.scalar_mediator import HiggsPortal
from hazma.vector_mediator import KineticMixing
from hazma.parameters import sv_inv_MeV_to_cm3_per_s
import hazma.gamma_ray_parameters as grp


this_dir = os.path.dirname(os.path.abspath(__file__))
head, _ = os.path.split(this_dir)
default_prefix = os.path.join(head, "results")


def sigmav(model: Union[HiggsPortal, KineticMixing], vx: float = 1e-3):
    """Compute <σv> for the given model."""
    cme = 2 * model.mx * (1.0 + 0.5 * vx ** 2)
    sig = model.annihilation_cross_sections(cme)["total"]
    return sig * vx * sv_inv_MeV_to_cm3_per_s


def gc_bg_model_approx():
    """
    Returns a background model simple to `hazma.gamma_ray_parameters.gc_bg_model`
    but it does not constrain the input energies.
    """
    return grp.BackgroundModel(
        [0, 1e5], lambda e: 7 * grp.default_bg_model.dPhi_dEdOmega(e)
    )


def make_parser(prog, description):
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
    )
    parser.add_argument("config", help="path to json file containing config.")
    return parser


def parse_json(args, requires, example_config):
    with open(args.config, "r") as f:
        config = json.load(f)

    required = ["filename", *requires]

    for key in required:
        if key not in config:
            print(f"Config is required to specify {key}. Example config:")
            example_config()
            raise ValueError()

    # These are common optional
    if "min-mx" not in config:
        config["min-mx"] = 0.1
    if "max-mx" not in config:
        config["max-mx"] = 250.0
    if "num-mx" not in config:
        config["num-mx"] = 100
    if "prefix" not in config:
        config["prefix"] = default_prefix
    if "sigma" not in config:
        config["sigma"] = 5.0
    if "overwrite" not in config:
        config["overwrite"] = False

    return config


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


def print_example_config(required: Dict, optional: Dict):
    console = Console()
    console.print("{")
    for key, val in required.items():
        console.print(f"[red bold]\t'{key}': {val}")
    for key, val in optional.items():
        console.print(f"\t'{key}': {val}")
    console.print("}")
