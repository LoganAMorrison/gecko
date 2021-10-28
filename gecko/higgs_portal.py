from typing import Optional, Tuple, Callable

import numpy as np
import numpy.typing as npt
from rich.progress import Progress
from hazma.scalar_mediator import HiggsPortal
from hazma.parameters import sv_inv_MeV_to_cm3_per_s

from . import iterators
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


class HiggsPortalPhenoConstrainer(AbstractConstrainer):
    """
    Class for computing the constraints on the dark-matter annihilation rate from pheno
    constraints in the HiggsPortal model.
    """

    def __init__(
        self,
        stheta: Callable[[float, float], float],
        vx: float = 1e-3,
        min_sv: float = 1e-40,
        max_sv: float = 1e-24,
        num_sv: int = 100,
        gsxx_max=4 * np.pi,
    ):
        super().__init__()
        self.stheta = stheta
        self.vx = vx
        self.svs = np.geomspace(min_sv, max_sv, num_sv)
        self.gsxx_max = gsxx_max

    @property
    def description(self) -> str:
        return "[medium_violet_red] Higgs Portal Pheno"

    @property
    def name(self):
        return "pheno"

    def _constrain(self, model):
        def f(sv):
            return self._pheno(model, sv)

        try:
            vals = np.vectorize(f)(self.svs)
            return self.svs[np.argmin(vals > 0)]
        except Exception as e:
            print(e)
            return np.nan

    def _pheno(self, model, sv):
        """
        Determines whether any values of stheta are consistent with pheno constraints
        at the given point in the ID plane.
        """
        mx = model.mx
        ms = model.ms
        vx = self.vx
        gsxx_max = self.gsxx_max

        assert ms > mx
        hp = HiggsPortal(mx, ms, 1, 1)
        sv_1 = (
            hp.annihilation_cross_sections(2 * hp.mx * (1 + 0.5 * vx ** 2))["total"]
            * vx
            * sv_inv_MeV_to_cm3_per_s
        )

        # Find smallest stheta compatible with <sigma v> for the given
        # gsxx_max
        stheta_min = np.sqrt(sv / sv_1) / gsxx_max
        if stheta_min > 0.999:
            return -1e100

        stheta_grid = np.geomspace(stheta_min, 0.999, 20)
        constr_mins = np.full_like(stheta_grid, np.inf)
        gsxxs = np.zeros_like(stheta_grid)
        for i, stheta in enumerate(stheta_grid):
            hp.stheta = stheta
            hp.gsxx = np.sqrt(sv / sv_1) / hp.stheta
            gsxxs[i] = hp.gsxx
            # Figure out strongest constraint
            constr_mins[i] = np.min([fn() for fn in hp.constraints().values()])

        # Check if (mx, ms, sv) point is allowed for some (gsxx, stheta) combination
        return constr_mins.max()


class HiggsPortalConstrainer(CompositeConstrainer):
    """
    Class for computing the constraints on the higgs-portal model
    from CMB, existing gamma-ray telescopes, the GECCO telescope and
    various ground-based experiments.
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
        return "[bold green] Higgs Portal"

    def _update_model(self, model: HiggsPortal, mx, ms):
        model.mx = mx
        model.ms = ms

    def _model_iterator(
        self,
        mxs: npt.NDArray[np.float64],
        ms_mx_ratio: float,
        gsxx: float = 1,
        stheta: float = 1.0,
    ):
        base = HiggsPortal(mx=1, ms=1, gsxx=gsxx, stheta=stheta)
        up = lambda model, mx: self._update_model(model, mx, mx * ms_mx_ratio)
        return iterators.ModelIterator(base).iterate(up, mxs)

    def constrain(
        self,
        mxs: npt.NDArray[np.float64],
        ms_mx_ratio: float = 2.0,
        cmb: bool = True,
        existing: bool = True,
        gecco: bool = True,
        pheno: bool = True,
        relic_density: bool = True,
        progress: Optional[Progress] = None,
        **options,
    ):
        """
        Compute the constraints on a Higgs-portal dark matter model. The observations
        used to constrain are: CMB, COMPTEL, EGRET, Fermi, INTEGRAL, GECCO and various
        ground-based experiments.

        Parameters
        ----------
        mxs: ArrayLike
            Array of dark-matter masses.
        ms_mx_ratio: float
            Ratio of scalar mass to dark-matter mass.
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
        stheta: float, optional
            Value of `stheta` to use in computing relic density. Default is 1.0.
        gsxx: Tuple[float,float], optional
            Upper and lower bounds of `gsxx` to use in computing relic density. Default
            is (1e-5, 4.0 * np.pi).

        Returns
        -------
        constraints: Dict
            Dictionary containing all constraints.

        Returns
        -------
        constraints: Dict
            Dictionary containing all constraints.
        """
        self.reset_constrainers()
        stheta = options.get("stheta", 1.0)
        gsxx: Tuple[float, float] = options.get("gsxx", (1e-5, 4.0 * np.pi))
        vx = options.get("vx", 1e-3)

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
            self.add_constrainer(
                HiggsPortalPhenoConstrainer(lambda _, ms: 5e-3 if ms > 400.0 else 4e-4)
            )

        if relic_density:
            self.add_constrainer(
                RelicDensityConstrainer("gsxx", gsxx[0], gsxx[1], vx=vx, log=True)
            )

        model_iterator = self._model_iterator(mxs, ms_mx_ratio, stheta=stheta)
        return self._constrain(model_iterator, progress)


def example_config():
    from .utils import print_example_config

    required = {
        "mass-ratio": "3.0",
        "filename": "'higgs_portal'",
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
        "higgs_portal", "Compute constraints on the Higgs-portal model."
    )

    config = parse_json(parser.parse_args(), ["mass-ratio"], example_config)

    mass_ratio = config["mass-ratio"]
    sigma = config["sigma"]
    max_mx = config["max-mx"]
    min_mx = config["min-mx"]
    num_mx = config["num-mx"]

    constrainer = HiggsPortalConstrainer()

    mxs = np.geomspace(min_mx, max_mx, num_mx)
    with Progress() as progress:
        constraints = constrainer.constrain(
            mxs, ms_mx_ratio=mass_ratio, progress=progress, sigma=sigma
        )
        constraints["masses"] = mxs

    write_data(config["prefix"], config["filename"], constraints, config["overwrite"])
