from typing import Optional
import os

import numpy as np
import numpy.typing as npt
from rich.progress import Progress
from scipy.interpolate import interp1d
from hazma.vector_mediator import KineticMixing

from . import iterators
from .utils import sigmav
from .constrainers import (
    AbstractConstrainer,
    RelicDensityConstrainer,
    CmbConstrainer,
    GeccoConstrainer,
    EgretConstrainer,
    ComptelConstrainer,
    FermiConstrainer,
    IntegralConstrainer,
    CompositeConstrainer,
)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class KineticMixingPhenoConstrainer(AbstractConstrainer):
    """
    Class for computing the 'phenomenological' constraints on the kinetic-mixing model
    from BaBar, LSND and E137.
    """

    def __init__(self, alphad: float = 0.5, vx: float = 1e-3):
        super().__init__()

        self._data = {
            "babar": np.genfromtxt(
                os.path.join(THIS_DIR, "data", "babar.csv"), delimiter=","
            ),
            "lsnd": np.genfromtxt(
                os.path.join(THIS_DIR, "data", "lsnd.csv"), delimiter=","
            ),
            "e137": np.genfromtxt(
                os.path.join(THIS_DIR, "data", "e137.csv"), delimiter=","
            ),
        }
        self._interp = {
            ex: interp1d(data.T[0], data.T[1]) for (ex, data) in self._data.items()
        }
        self.alphad = alphad
        self.vx = vx

    @property
    def description(self) -> str:
        return "[medium_violet_red] Kinetic Mixing Pheno"

    @property
    def name(self):
        return "pheno"

    def _constrain_single(self, model, ex):
        mx = model.mx
        mv = model.mv
        if mx * 1e-3 > self._data[ex].T[0][-1]:
            return np.inf
        if mx * 1e-3 < self._data[ex].T[0][0]:
            y = self._data[ex].T[1][0]
        else:
            y = self._interp[ex](mx * 1e-3)
        gvxx = np.sqrt(4.0 * np.pi * self.alphad)
        eps = np.sqrt((mx / mv) ** 4 * y / self.alphad)
        model = KineticMixing(**{"mx": mx, "mv": 3.0 * mx, "gvxx": gvxx, "eps": eps})
        return sigmav(model, self.vx)

    def _constrain(self, model):
        return np.min([self._constrain_single(model, ex) for ex in self._data.keys()])


class KineticMixingConstrainer(CompositeConstrainer):
    """
    Class for computing the constraints on the kinetic-mixing model
    from CMB, existing gamma-ray telescopes, the GECCO telescope and
    BaBar, LSND and E137.
    """

    def __init__(self):
        super().__init__()
        self._gecco_targets = GeccoConstrainer.annihilation_targets()

    def gecco_targets(self):
        """
        Return the GECCO targets used.
        """
        return self._gecco_targets

    @property
    def description(self) -> str:
        return "[bold red] Kinetic Mixing"

    def _update(self, model: KineticMixing, mx, mv):
        model.mx = mx
        model.mv = mv

    def _model_iterator(
        self,
        mxs: npt.NDArray[np.float64],
        mv_mx_ratio: float,
        gvxx: float = 1.0,
        eps: float = 1e-3,
    ):
        base = KineticMixing(mx=1, mv=1, gvxx=gvxx, eps=eps)
        up = lambda model, mx: self._update(model, mx, mx * mv_mx_ratio)
        return iterators.ModelIterator(base).iterate(up, mxs)

    def constrain(
        self,
        mxs: npt.NDArray[np.float64],
        mv_mx_ratio: float = 3.0,
        cmb: bool = True,
        existing: bool = True,
        gecco: bool = True,
        pheno: bool = True,
        relic_density: bool = True,
        progress: Optional[Progress] = None,
        **options,
    ):
        """
        Compute the constraints on a kinetic-mixing dark matter model. The observations
        used to constrain are: CMB, COMPTEL, EGRET, Fermi, INTEGRAL and GECCO.

        Parameters
        ----------
        mxs: ArrayLike
            Array of dark-matter masses.
        mv_mx_ratio: float, optional
            Ratio of vector mass to dark-matter mass. Default is 3.0.
        cmb: bool, optional
            If true, CMB constraints are computed. Default is True.
        existing: bool, optional
            If true, constraints from existing telescopes are computed. Default is True.
        gecco: bool, optional
            If true, constraints from GECCO are computed. Default is True.
        pheno: bool, optional
            If true, pheno constraints are computed. Default is True.
        relic_density: bool, optional
            If true, relic-density constraints are computed. Default is True.
        progress: rich.progress.Progress or None, optional
            Rich progress-bar to display progress.
        options
            Options to pass to various constrainers. These options are listed below.
        sigma: float, optional
            Discovery threshold for GECCO corresponding to a singal-to-noise ratio
            greater than `sigma`. Default is 5.0.
        vx: float, optional
            Dark-matter velocity today. Default is 1e-3.
        gvxx: float, optional
            Value of `gvxx` to use in computing relic density. Default is 1.0.
        eps: Tuple[float,float], optional
            Upper and lower values of `eps` to use in computing relic density. Default
            is (1e-10, 1.0).

        Returns
        -------
        constraints: Dict
            Dictionary containing all constraints.
        """
        self.reset_constrainers()
        gvxx = options.get("gvxx", 1.0)
        eps = options.get("eps", (1e-10, 1.0))
        vx = options.get("vx", 1e-3)
        alphad = options.get("alphad", 0.5)

        if cmb:
            self.add_constrainer(CmbConstrainer())
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

        if pheno:
            self.add_constrainer(KineticMixingPhenoConstrainer(alphad, vx))

        if relic_density:
            self.add_constrainer(
                RelicDensityConstrainer("eps", eps[0], eps[1], vx=vx, log=True)
            )

        model_iterator = self._model_iterator(mxs, mv_mx_ratio, gvxx=gvxx)
        return self._constrain(model_iterator, progress)


def example_config():
    from .utils import print_example_config

    required = {
        "mass-ratio": "3.0",
        "filename": "'kinetic_mixing'",
    }
    optional = {
        "min-mx": 0.1,
        "max-mx": 250.0,
        "num-mx": 100,
        "sigma": [5.0, 25.0],
        "overwrite": "false",
    }
    print_example_config(required, optional)


if __name__ == "__main__":
    from .utils import parse_json, make_parser, write_data

    parser = make_parser(
        "kinetic-mixing", "Compute constraints on the Kinetic Mixing model."
    )

    config = parse_json(parser.parse_args(), ["mass-ratio"], example_config)

    constrainer = KineticMixingConstrainer()
    mass_ratio = config["mass-ratio"]
    sigma = config["sigma"]
    max_mx = config["max-mx"]
    min_mx = config["min-mx"]
    num_mx = config["num-mx"]

    mxs = np.geomspace(min_mx, max_mx, num_mx)
    with Progress() as progress:
        constraints = constrainer.constrain(
            mxs, mass_ratio, progress=progress, sigma=sigma
        )
        constraints["masses"] = mxs

    write_data(config["prefix"], config["filename"], constraints, config["overwrite"])
