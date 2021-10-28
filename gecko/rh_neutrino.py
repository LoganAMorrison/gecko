from typing import Optional

import numpy as np
import numpy.typing as npt
from rich.progress import Progress
from hazma.rh_neutrino import RHNeutrino


from . import iterators
from .constrainers import (
    GeccoConstrainer,
    EgretConstrainer,
    ComptelConstrainer,
    FermiConstrainer,
    IntegralConstrainer,
    CompositeConstrainer,
)


class RHNeutrinoConstrainer(CompositeConstrainer):
    """
    Class for computing the constraints on the RH-neutrino model
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
        return "[bold blue] RH-Neutrino"

    def _update(self, model: RHNeutrino, mx):
        model.mx = mx

    def _model_iterator(
        self,
        mxs: npt.NDArray[np.float64],
        stheta: float = 1e-3,
        lepton: str = "e",
    ):
        base = RHNeutrino(1, stheta, lepton=lepton, include_3body=True)
        up = lambda model, mx: self._update(model, mx)
        return iterators.ModelIterator(base).iterate(up, mxs)

    def constrain(
        self,
        mxs: npt.NDArray[np.float64],
        existing: bool = True,
        gecco: bool = True,
        progress: Optional[Progress] = None,
        **options,
    ):
        """
        Compute the constraints on a RH-Neutrino dark matter model. The observations
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
        lepton: str, optional
            Lepton flavor the RH-neutrino mixes with. Can be "e" or "mu".
            Default is "e".

        Returns
        -------
        constraints: Dict
            Dictionary containing all constraints.
        """
        self.reset_constrainers()

        stheta = options.get("stheta", 1e-3)
        lepton = options.get("lepton", "e")
        model_iterator = self._model_iterator(mxs, stheta=stheta, lepton=lepton)

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

        return self._constrain(model_iterator, progress)


def example_config():
    from .utils import print_example_config

    required = {
        "lepton": "'e'",
        "filename": "'rhn'",
    }
    optional = {
        "mx-min": 0.1,
        "mx-max": 500.0,
        "num-mx": 100,
        "sigma": 5.0,
        "overwrite": "false",
    }
    print_example_config(required, optional)


if __name__ == "__main__":
    from .utils import parse_json, make_parser, write_data

    parser = make_parser(
        "kinetic-mixing", "Compute constraints on the Kinetic Mixing model."
    )

    config = parse_json(parser.parse_args(), ["lepton"], example_config)
    lepton = config["lepton"]
    sigma = config["sigma"]
    max_mx = config["max-mx"]
    min_mx = config["min-mx"]
    num_mx = config["num-mx"]

    rhn = RHNeutrinoConstrainer()

    mxs = np.geomspace(min_mx, max_mx, num_mx)
    with Progress() as progress:
        constraints = rhn.constrain(mxs, lepton=lepton, progress=progress, sigma=sigma)
        constraints["masses"] = mxs

    write_data(config["prefix"], config["filename"], constraints, config["overwrite"])
