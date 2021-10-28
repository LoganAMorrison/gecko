from typing import Optional

import numpy as np
from rich.progress import Progress
from hazma.parameters import g_to_MeV
from hazma.pbh import PBH

from . import iterators
from .constrainers import (
    GeccoConstrainer,
    EgretConstrainer,
    ComptelConstrainer,
    FermiConstrainer,
    IntegralConstrainer,
    CompositeConstrainer,
)


class PbhConstrainer(CompositeConstrainer):
    """
    Class for computing the constraints on the primortial black holes
    from existing gamma-ray telescopes and the GECCO telescope.
    """

    def __init__(self):
        super().__init__()
        self._gecco_targets = GeccoConstrainer.decay_targets()

    def gecco_targets(self):
        """
        Return the GECCO targets used.
        """
        return self._gecco_targets

    @property
    def description(self) -> str:
        return "[bold magenta] PBH"

    def _update(self, model: PBH, mx):
        model.mx = mx

    def _model_iterator(self, spectrum_kind: str = "secondary"):
        base = PBH(1e15 * g_to_MeV, spectrum_kind=spectrum_kind)
        up = lambda model, mx: self._update(model, mx)
        return iterators.ModelIterator(base).iterate(up, base._mxs)

    def constrain(
        self,
        existing: bool = True,
        gecco: bool = True,
        progress: Optional[Progress] = None,
        **options,
    ):
        """
        Compute the constraints on a PBH model. The observations
        used to constrain are: COMPTEL, EGRET, Fermi, INTEGRAL and GECCO.

        Parameters
        ----------
        mxs: ArrayLike
            Array of dark-matter masses.
        existing: bool, optional
            If true, constraints from existing telescopes are computed. Default is True.
        gecco: bool, optional
            If true, constraints from GECCO are computed. Default is True.
        progress: rich.progress.Progress or None, optional
            Rich progress-bar to display progress.
        options
            Options to pass to various constrainers. These options are listed below.
        sigma: float, optional
            Discovery threshold for GECCO corresponding to a singal-to-noise ratio
            greater than `sigma`. Default is 5.0.
        spectrum_kind: str, optional
            Kind of spectrum to include from PBH evaporation. Default is 'secondary'
        include_masses: bool, optional
            If true, the masses used to generate constrains are included in the output.
            Default is True.

        Returns
        -------
        constraints: Dict
            Dictionary containing all constraints.
        """
        spectrum_kind = options.get("spectrum_kind", "secondary")
        model_iterator = self._model_iterator(spectrum_kind=spectrum_kind)

        if existing:
            self.add_constrainer(EgretConstrainer())
            self.add_constrainer(ComptelConstrainer())
            self.add_constrainer(IntegralConstrainer())
            self.add_constrainer(FermiConstrainer())
        if gecco:
            for key, (target, bgmodel) in self._gecco_targets.items():
                sigma = options.get("sigma", 5.0)
                tobs = options.get("tobs", 1e6)
                self.add_constrainer(
                    GeccoConstrainer(key, bgmodel, target, tobs, sigma)
                )

        constraints = self._constrain(model_iterator, progress)

        if options.get("include_masses", True):
            constraints["masses"] = np.array([model.mx for model in model_iterator])

        return constraints


def example_config():
    from .utils import print_example_config

    required = {
        "filename": "'pbh'",
    }
    optional = {
        "sigma": [5.0, 25.0],
        "overwrite": "false",
    }
    print_example_config(required, optional)


if __name__ == "__main__":
    from .utils import parse_json, make_parser, write_data

    parser = make_parser("pbh", "Compute constraints on the primordial black holes.")
    config = parse_json(parser.parse_args(), [], example_config)
    sigma = config["sigma"]

    constrainer = PbhConstrainer()

    with Progress() as progress:
        constraints = constrainer.constrain(
            progress=progress, sigma=sigma, include_masses=True
        )

    write_data(config["prefix"], config["filename"], constraints, config["overwrite"])
