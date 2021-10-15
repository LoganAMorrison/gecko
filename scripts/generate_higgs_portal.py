# -*- python -*-

import numpy as np
from rich.progress import Progress
import argparse

from utils import write_data, parse_json, base_parser

try:
    from gecko import HiggsPortalConstraints
except ImportError:
    import sys

    sys.path.append("..")
    from gecko import HiggsPortalConstraints


def generate(prefix, filename, mx_min, mx_max, mass_ratio, sigmas, overwrite):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_higgs_portal",
        description="Compute constraints on the Higgs-portal model.",
        parents=[base_parser],
    )

    extra_required = ["mass-ratio"]
    extra_optional = {"mx-min": 0.1, "mx-max": 250.0}
    config = parse_json(parser.parse_args(), extra_required, extra_optional)

    generate(
        config["prefix"],
        config["filename"],
        config["mx-min"],
        config["mx-max"],
        config["mass-ratio"],
        config["sigma"],
        config["overwrite"],
    )
