from typing import Callable, Dict, Iterable, Optional, Tuple, Union
import os

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d
import hazma.gamma_ray_parameters as _grp
import hazma.single_channel as _single_channel
from hazma.vector_mediator import KineticMixing
from hazma.scalar_mediator import HiggsPortal
from hazma.rh_neutrino import RHNeutrino
from rich.progress import Progress, TaskID


from . import iterators
from .utils import sigmav
from .constrainers import (
    AbstractConstrainer,
    NewTelescopeConstrainer,
    ExistingTelescopeConstrainer,
    CmbConstrainer,
    RelicDensityConstrainer,
)
from .gecco import (
    decay_targets as gecco_decay_targets,
    annihilation_targets as gecco_annihilation_targets,
    effective_area as gecco_effective_area,
    energy_resolution as gecco_energy_resolution,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

ConstrainerType = Union[
    ExistingTelescopeConstrainer,
    NewTelescopeConstrainer,
    CmbConstrainer,
    RelicDensityConstrainer,
]

ConstrainerDict = Union[
    Dict[str, ExistingTelescopeConstrainer],
    Dict[str, NewTelescopeConstrainer],
    Dict[str, CmbConstrainer],
    Dict[str, RelicDensityConstrainer],
]


def gecco_constrainer(background_model, target, observation_time, sigma):
    """
    Return a constrainer for the GECCO telescope.
    """
    return NewTelescopeConstrainer(
        gecco_effective_area(),
        gecco_energy_resolution(),
        target,
        background_model,
        observation_time,
        sigma,
    )


def comptel_constrainer(sigma=2.0, method="1bin"):
    """
    Return a constrainer for the COMPTEL telescope.
    """
    return ExistingTelescopeConstrainer(_grp.comptel_diffuse, sigma, method)


def egret_constrainer(sigma=2.0, method="1bin"):
    """
    Return a constrainer for the EGRET telescope.
    """
    return ExistingTelescopeConstrainer(_grp.egret_diffuse, sigma, method)


def fermi_constrainer(sigma=2.0, method="1bin"):
    """
    Return a constrainer for the Fermi telescope.
    """
    return ExistingTelescopeConstrainer(_grp.fermi_diffuse, sigma, method)


def integral_constrainer(sigma=2.0, method="1bin"):
    """
    Return a constrainer for the INTEGRAL telescope.
    """
    return ExistingTelescopeConstrainer(_grp.integral_diffuse, sigma, method)


def _constrain(
    model_iterator,
    constrainers: ConstrainerDict,
    progress: Optional[Tuple[Progress, TaskID]] = None,
    **options,
):
    total = len(model_iterator)
    constraints = {}
    color = options.get("color", "red")

    for key, constrainer in constrainers.items():
        if progress is not None:
            task = (
                progress[0],
                progress[0].add_task(f"[{color}] {key}", total=total, refresh=True),
            )
        else:
            task = None
        constraints[key] = constrainer.constrain(model_iterator, task)

        if progress is not None:
            progress[0].update(progress[1], advance=1, refresh=True)

    return constraints


def astro_constraints(
    model_iterator: Iterable,
    gecco_targets: Dict,
    cmb: bool = True,
    existing: bool = True,
    gecco: bool = True,
    progress_task: Optional[Tuple[Progress, TaskID]] = None,
    **options,
):
    """
    Compute the constraints from CMB, COMPTEL, EGRET, FERMI, INTEGRAL and GECCO
    for each model in the iterator.

    Parameters
    ----------
    model_iterator: Iterable
        Iterator over the models to constrain.
    gecco_targets: Dict
        Dictionary where the values are pairs of BackgroundModel and TargetParams.
    cmb: bool
        If true, CMB constraints are computed. Default is True.
    existing: bool
        If true, constraints from existing telescopes are computed. Default is True.
    gecco: bool = True,
        If true, constraints from GECCO are computed. Default is True.
    options
        Options to pass to various constrainers. These options are listed below.
    sigma: float, optional
        Discovery threshold for GECCO corresponding to a singal-to-noise ratio
        greater than `sigma`. Default is 5.0.

    Returns
    -------
    constraints: Dict
        Dictionary containing all constraints. The format is:
            constraints = {
                "cmb": [...],
                "egret": [...],
                ...
                "integral": [...],
                gecco_target1: {
                    "nt_sigma[0] sigma": [...],
                    ...
                    "nt_sigma[-1] sigma": [...],
                }
            }
    """
    total = 0
    if cmb:
        total += 1
        cmb_constrainer = CmbConstrainer()
    else:
        cmb_constrainer = None

    if existing:
        total += 4
        existing_constrainers = {
            "comptel": comptel_constrainer(),
            "egret": egret_constrainer(),
            "fermi": fermi_constrainer(),
            "integral": integral_constrainer(),
        }
    else:
        existing_constrainers = None

    if gecco:
        total += len(gecco_targets)
        sigma = options.get("sigma", 5.0)
        tobs = options.get("tobs", 1e6)
        gecco_constrainers = {
            name: gecco_constrainer(bgmodel, target, tobs, sigma)
            for name, (bgmodel, target) in gecco_targets.items()
        }
    else:
        gecco_constrainers = None

    if progress_task is not None:
        progress_task[0].update(progress_task[1], total=total)
        progress = (progress_task[0], progress_task[1])
    else:
        progress = None

    constraints = {}
    if cmb_constrainer is not None:
        constraints = {
            **constraints,
            **_constrain(
                model_iterator, {"cmb": cmb_constrainer}, progress, color="blue"
            ),
        }

    if existing_constrainers is not None:
        constraints = {
            **constraints,
            **_constrain(
                model_iterator, existing_constrainers, progress, color="purple"
            ),
        }

    if gecco_constrainers is not None:
        constraints = {
            **constraints,
            **_constrain(model_iterator, gecco_constrainers, progress, color="green"),
        }

    return constraints


class SingleChannelConstraints:
    """
    Class for computing the constraints on the dark-matter annihilation rate assuming
    the dark-matter annihilates/decay into a single final state.
    """

    def __init__(self):
        pass

    def update(self, model, mx):
        model.mx = mx

    def model_iterator(self, mxs, finalstate, decay):
        if decay:
            base = _single_channel.SingleChannelDec(0.0, finalstate, 1.0)
        else:
            base = _single_channel.SingleChannelAnn(0.0, finalstate, 1.0)
        up = lambda model, mx: self.update(model, mx)
        return iterators.ModelIterator(base).iterate(up, mxs)

    def compute(
        self,
        mxs: npt.NDArray[np.float64],
        finalstate: str,
        decay: bool,
        cmb: bool = True,
        existing: bool = True,
        gecco: bool = True,
        progress: Optional[Progress] = None,
        **options,
    ):
        """
        Compute the constraints on simple annihilation/decay models
        decaying/annihilating only into pairs of: electrons, muons, pions and photons.
        The observations used to constrain are: CMB, COMPTEL, EGRET, Fermi, INTEGRAL
        and GECCO.

        Parameters
        ----------
        mxs: ArrayLike
            Array of dark-matter masses.
        finalstate: str
            Final state the dark-matter decay/annihilates into.
        decay: bool
            If true, the dark-matter is assumed to decay into the final state.
            Otherwise, annihilate into the final state.
        cmb: bool
            If true, CMB constraints are computed. Default is True.
        existing: bool
            If true, constraints from existing telescopes are computed. Default is True.
        gecco: bool = True,
            If true, constraints from GECCO are computed. Default is True.
        progress: Optional[Progress]
            Rich progress-bar to display progress.
        options
            Options to pass to various constrainers. These options are listed below.
        sigma: float
            Discovery threshold for GECCO corresponding to a singal-to-noise ratio
            greater than `sigma`. Default is 5.0.

        Returns
        -------
        constraints: Dict
            Dictionary containing all constraints. The format is:
            constraints = { "ann": ann_constraints, "dec": dec_constraints} where the
            annihilation and decay constraints contain constrains from each observation.
        """
        gecco_targets = gecco_decay_targets() if decay else gecco_annihilation_targets()
        # final_states = ["e e", "g g", "mu mu", "pi pi"]
        model_iterator = self.model_iterator(mxs, finalstate, decay)

        if progress is not None:
            ty = "Decay" if decay else "Annihilation"
            desc = f"[red bold]{ty} -> {finalstate}"
            task = progress.add_task(desc, total=len(mxs))
            progress_task = (progress, task)
        else:
            progress_task = None

        return astro_constraints(
            model_iterator,
            gecco_targets,
            cmb,
            existing,
            gecco,
            progress_task,
            **options,
        )


class HiggsPortalPhenoConstrainer(AbstractConstrainer):
    """
    Class for computing the constraints on the dark-matter annihilation rate from pheno
    constraints in the HiggsPortal model.
    """

    def __init__(self, stheta: Callable[[float, float], float], vx: float = 1e-3):
        super().__init__()
        self.stheta = stheta
        self.vx = vx

    def _constrain(self, model):
        model.stheta = self.stheta(model.mx, model.ms)
        return sigmav(model, self.vx)


class HiggsPortalConstraints:
    """
    Class for computing the constraints on the higgs-portal model
    from CMB, existing gamma-ray telescopes, the GECCO telescope and
    various ground-based experiments.
    """

    def __init__(self):
        self._gecco_targets = gecco_annihilation_targets()
        self._task_names = ["telescopes", "pheno", "relic-density"]

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

    def _progress_task(
        self, desc: str, progress: Optional[Progress] = None, total: int = 0
    ):
        if progress is not None:
            header = f"higgs-portal: {desc}"
            progress_task = (
                progress,
                progress.add_task(header, total=total),
            )
        else:
            progress_task = None
        return progress_task

    def compute(
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
        model_iterator = self._model_iterator(mxs, ms_mx_ratio)
        constraints = {}

        if cmb or existing or gecco:
            progress_task = self._progress_task("telescopes", progress, len(mxs))
            constraints = astro_constraints(
                model_iterator,
                self._gecco_targets,
                cmb,
                existing,
                gecco,
                progress_task,
                **options,
            )
        if pheno:
            pheno_constrainer = HiggsPortalPhenoConstrainer(
                lambda _, ms: 5e-3 if ms > 400.0 else 4e-4
            )
            progress_task = self._progress_task("pheno", progress, len(mxs))
            constraints["pheno"] = pheno_constrainer.constrain(
                model_iterator, progress_task
            )

        if relic_density:
            stheta = options.get("stheta", 1.0)
            gsxx: Tuple[float, float] = options.get("gsxx", (1e-5, 4.0 * np.pi))
            vx = options.get("vx", 1e-3)
            model_iterator = self._model_iterator(mxs, ms_mx_ratio, stheta=stheta)
            progress_task = self._progress_task("relic-density", progress, len(mxs))
            rd_constrainer = RelicDensityConstrainer(
                "gsxx", gsxx[0], gsxx[1], vx=vx, log=True
            )
            constraints["relic-density"] = rd_constrainer.constrain(
                model_iterator, progress_task
            )
        return constraints


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


class KineticMixingConstraints:
    """
    Class for computing the constraints on the kinetic-mixing model
    from CMB, existing gamma-ray telescopes, the GECCO telescope and
    BaBar, LSND and E137.
    """

    def __init__(self):
        self._gecco_targets = gecco_annihilation_targets()
        self._task_names = ["telescopes", "pheno", "relic-density"]

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

    def _progress_task(
        self, desc: str, progress: Optional[Progress] = None, total: int = 0
    ):
        if progress is not None:
            header = f"kinetic-mixing: {desc}"
            progress_task = (
                progress,
                progress.add_task(header, total=total),
            )
        else:
            progress_task = None
        return progress_task

    def compute(
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
        model_iterator = self._model_iterator(mxs, mv_mx_ratio)
        constraints = {}

        if cmb or existing or gecco:
            progress_task = self._progress_task("telescopes", progress, len(mxs))
            constraints = astro_constraints(
                model_iterator,
                self._gecco_targets,
                cmb,
                existing,
                gecco,
                progress_task,
                **options,
            )
        if pheno:
            alphad = options.get("alphad", 0.5)
            vx = options.get("vx", 1e-3)
            pheno_constrainer = KineticMixingPhenoConstrainer(alphad, vx)
            progress_task = self._progress_task("pheno", progress, len(mxs))
            constraints["pheno"] = pheno_constrainer.constrain(
                model_iterator, progress_task
            )

        if relic_density:
            gvxx = options.get("gvxx", 1.0)
            eps = options.get("eps", (1e-10, 1.0))
            vx = options.get("vx", 1e-3)
            model_iterator = self._model_iterator(mxs, mv_mx_ratio, gvxx=gvxx)
            progress_task = self._progress_task("relic-density", progress, len(mxs))
            rd_constrainer = RelicDensityConstrainer(
                "eps", eps[0], eps[1], vx=vx, log=True
            )
            constraints["relic-density"] = rd_constrainer.constrain(
                model_iterator, progress_task
            )

        return constraints


class RHNeutrinoConstraints:
    """
    Class for computing the constraints on the RH-neutrino model
    from existing gamma-ray telescopes and the GECCO telescope.
    """

    def __init__(self):
        self._gecco_targets = gecco_decay_targets()
        self._task_names = ["telescopes"]

    def _update(self, model: RHNeutrino, mx):
        model.mx = mx

    def _model_iterator(
        self,
        mxs: npt.NDArray[np.float64],
        stheta: float = 1e-3,
        lepton: str = "e",
    ):
        base = RHNeutrino(1, stheta, lepton=lepton)
        up = lambda model, mx: self._update(model, mx)
        return iterators.ModelIterator(base).iterate(up, mxs)

    def _progress_task(
        self, desc: str, progress: Optional[Progress] = None, total: int = 0
    ):
        if progress is not None:
            header = f"RH-Neutrino {desc}"
            progress_task = (
                progress,
                progress.add_task(header, total=total),
            )
        else:
            progress_task = None
        return progress_task

    def compute(
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
        stheta = options.get("stheta", 1e-3)
        lepton = options.get("lepton", "e")
        model_iterator = self._model_iterator(mxs, stheta=stheta, lepton=lepton)
        constraints = {}

        if existing or gecco:
            progress_task = self._progress_task("telescopes", progress, len(mxs))
            constraints = astro_constraints(
                model_iterator,
                self._gecco_targets,
                cmb=False,
                existing=existing,
                gecco=gecco,
                progress_task=progress_task,
                **options,
            )

        return constraints
