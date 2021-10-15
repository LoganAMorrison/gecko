# -*- python -*-

import numpy as np
from rich.progress import Progress
import argparse
from hazma.parameters import (
    electron_mass as me,
    muon_mass as mmu,
    charged_pion_mass as mpi,
)

from utils import write_data, parse_json, base_parser

try:
    from gecko import SingleChannelConstraints
except ImportError:
    import sys

    sys.path.append("..")
    from gecko import SingleChannelConstraints


def generate(prefix, filename, sigmas, overwrite):
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
            constraints[fs] = {"masses": mxs}
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

    write_data(prefix, filename, constraints, overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_single_channel_ann",
        description="Compute constraints on the single-channel annihilation model.",
        parents=[base_parser],
    )

    config = parse_json(parser.parse_args())

    generate(
        config["prefix"],
        config["filename"],
        config["sigma"],
        config["overwrite"],
    )
