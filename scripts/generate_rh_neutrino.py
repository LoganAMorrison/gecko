# -*- python -*-

import numpy as np
from rich.progress import Progress
import argparse

from utils import write_data, parse_json, base_parser

try:
    from gecko import RHNeutrinoConstraints
except ImportError:
    import sys

    sys.path.append("..")
    from gecko import RHNeutrinoConstraints


def generate(prefix, filename, mx_min, mx_max, lepton, sigmas, overwrite):
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
        prog="generate_rh_neutrino",
        description="Compute constraints on the kinetic-mixing model.",
        parents=[base_parser],
    )

    extra_required = ["lepton"]
    extra_optional = {"mx-min": 0.1, "mx-max": 250.0}
    config = parse_json(parser.parse_args(), extra_required, extra_optional)

    generate(
        config["prefix"],
        config["filename"],
        config["mx-min"],
        config["mx-max"],
        config["lepton"],
        config["sigma"],
        config["overwrite"],
    )
