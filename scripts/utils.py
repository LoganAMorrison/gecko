import argparse
import os
from pathlib import Path
from typing import Optional, Dict, List
import json

import h5py

this_dir = os.path.dirname(os.path.abspath(__file__))
head, _ = os.path.split(this_dir)
default_prefix = os.path.join(head, "results")

base_parser = argparse.ArgumentParser(add_help=False)
base_parser.add_argument("config", help="path to json file containing config.")


def parse_json(
    args,
    extra_required: Optional[List[str]] = None,
    extra_optional: Optional[Dict] = None,
):
    with open(args.config, "r") as f:
        config = json.load(f)
    # These are required
    conf = {"filename": config["filename"]}
    # These are optional
    conf["prefix"] = config.get("prefix", default_prefix)
    conf["filename"] = config.get("filename", "higgs_portal")
    conf["sigma"] = config.get("sigma", [5.0, 25.0])
    conf["overwrite"] = config.get("overwrite", False)
    if extra_required is not None:
        for key in extra_required:
            if key not in config:
                raise ValueError(f"Config is required to have {key}.")
            else:
                conf[key] = config[key]
    if extra_optional is not None:
        for key, val in extra_optional.items():
            conf[key] = config.get(key, val)
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
