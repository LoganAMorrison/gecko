import warnings
from copy import copy
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import root_scalar, RootResults
from typing import Optional
from rich.progress import Progress

from hazma.relic_density import relic_density
from hazma.parameters import omega_h2_cdm
import hazma.gamma_ray_parameters as grp

from .utils import sigmav, gc_bg_model_approx

# ============================================================================
# ---- Abstract Constrainers -------------------------------------------------
# ============================================================================


class AbstractConstrainer(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def _constrain(self, model):
        pass

    def constrain(self, model_iterator, progress: Optional[Progress] = None):
        """
        Compute the constraints on the models.

        Parameters
        ----------
        model_iterator: iter
            Iterator over the dark matter models.
        progress: Optional[Progress]
            A `rich` progress object to track progress.

        Returns
        -------
        constraints: array-like
            Numpy array containing the constraints for each model.
        """
        if progress is None:
            progress_update = lambda: None
        else:
            task = progress.add_task(self.description, total=len(model_iterator))
            progress_update = lambda: progress.update(task, advance=1, refresh=True)

        constraints = np.zeros((len(model_iterator),), dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, model in enumerate(model_iterator):
                constraints[i] = self._constrain(model)
                progress_update()
        return constraints


class CompositeConstrainer(ABC):
    def __init__(self, constrainers=[]):
        self._constrainers = constrainers

    def add_constrainer(self, constrainer):
        self._constrainers.append(constrainer)

    def reset_constrainers(self):
        self._constrainers = []

    @property
    @abstractmethod
    def description(self) -> str:
        raise NotImplementedError()

    def __len__(self):
        return len(self._constrainers)

    def _constrain(self, model_iterator, progress: Optional[Progress] = None):
        if progress is None:
            overall = None
        else:
            overall = progress.add_task(self.description, total=len(self))

        constraints = {}
        for constrainer in self._constrainers:
            name = constrainer.name
            constraints[name] = constrainer.constrain(model_iterator, progress)

            if progress is not None and overall is not None:
                progress.update(overall, advance=1, refresh=True)

        return constraints


class NewTelescopeConstrainer(AbstractConstrainer):
    """
    Class of computing the constraints on the dark-matter models from a new telescope.
    """

    def __init__(
        self,
        effective_area,
        energy_resolution,
        background_model,
        target,
        observation_time,
        sigma,
    ):
        super().__init__()
        self.effective_area = effective_area
        self.energy_resolution = energy_resolution
        self.background_model = background_model
        self.target = target
        self.observation_time = observation_time
        self.sigma = sigma

    def _constrain(self, model):
        return model.unbinned_limit(
            self.effective_area,
            self.energy_resolution,
            self.observation_time,
            self.target,
            self.background_model,
            n_sigma=self.sigma,
        )


class ExistingTelescopeConstrainer(AbstractConstrainer):
    """
    Class for computing the constraints on the dark-matter models from an
    existing telescope.
    """

    def __init__(self, measurement, sigma=2.0, method="1bin"):
        super().__init__()
        self.measurement = measurement
        self.sigma = sigma
        self.method = method

    def _constrain(self, model):
        return model.binned_limit(self.measurement, self.sigma, self.method)


# ============================================================================
# ---- General Constrainers --------------------------------------------------
# ============================================================================


class CmbConstrainer(AbstractConstrainer):
    """
    Class for computing constraints on dark-matter models from CMB.
    """

    def __init__(self, x_kd=1e-6):
        super().__init__()
        self.x_kd = x_kd

    def _constrain(self, model):
        return model.cmb_limit(x_kd=self.x_kd)

    @property
    def description(self):
        return "[purple] CMB"

    @property
    def name(self):
        return "cmb"


class RelicDensityConstrainer(AbstractConstrainer):
    """
    Class for computing constraints on dark-matter models from relic-density.
    """

    def __init__(
        self,
        prop: str,
        prop_min: float,
        prop_max: float,
        vx: float = 1e-3,
        log: bool = True,
    ):
        """
        Create a constrainer object for constraining the dark-matter
        annihilation cross section by varying a specified property
        such that the model yields the correct relic-density.

        Parameters
        ----------
        prop: str
            String specifying the property to vary in order fix the
            dark-matter relic-density.
        prop_min: float
            Minimum value of the property.
        prop_max: float
            Maximum value of the property.
        vx: float, optional
            The dark-matter velocity used to compute the annihilation cross
            section. Default is 1e-3.
        log: bool, optional
            If true, the property is varied logarithmically.
        """
        super().__init__()
        self.prop = prop
        self.prop_min = prop_min
        self.prop_max = prop_max
        self.vx = vx
        self.log = log

    @property
    def description(self):
        return "[dark_violet] Relic Density"

    @property
    def name(self):
        return "relic-density"

    def _setprop(self, model, val):
        if self.log:
            setattr(model, self.prop, 10 ** val)
        else:
            setattr(model, self.prop, val)

    def _constrain(self, model):
        model_ = copy(model)
        lb = self.prop_min if not self.log else np.log10(self.prop_min)
        ub = self.prop_max if not self.log else np.log10(self.prop_max)

        def f(val):
            self._setprop(model_, val)
            return relic_density(model_, semi_analytic=True) - omega_h2_cdm

        try:
            root: RootResults = root_scalar(f, bracket=[lb, ub], method="brentq")
            if not root.converged:
                warnings.warn(f"root_scalar did not converge. Flag: {root.flag}")
            self._setprop(model_, root.root)
            return sigmav(model_, self.vx)
        except ValueError as e:
            warnings.warn(f"Error encountered: {e}. Returning nan", RuntimeWarning)
            return np.nan


class ComptelConstrainer(ExistingTelescopeConstrainer):
    def __init__(self, sigma=2.0, method="1bin"):
        super().__init__(grp.comptel_diffuse, sigma, method)

    @property
    def description(self):
        return "[deep_sky_blue2] COMPTEL"

    @property
    def name(self):
        return "comptel"


class EgretConstrainer(ExistingTelescopeConstrainer):
    def __init__(self, sigma=2.0, method="1bin"):
        super().__init__(grp.egret_diffuse, sigma, method)

    @property
    def description(self):
        return "[deep_sky_blue1] EGRET"

    @property
    def name(self):
        return "egret"


class FermiConstrainer(ExistingTelescopeConstrainer):
    def __init__(self, sigma=2.0, method="1bin"):
        super().__init__(grp.fermi_diffuse, sigma, method)

    @property
    def description(self):
        return "[light_sea_green] Fermi"

    @property
    def name(self):
        return "fermi"


class IntegralConstrainer(ExistingTelescopeConstrainer):
    def __init__(self, sigma=2.0, method="1bin"):
        super().__init__(grp.integral_diffuse, sigma, method)

    @property
    def description(self):
        return "[dark_cyan] INTEGRAL"

    @property
    def name(self):
        return "integral"


class GeccoConstrainer(NewTelescopeConstrainer):
    def __init__(self, name, background_model, target, observation_time, sigma):
        self._name = name
        super().__init__(
            grp.effective_area_gecco,
            grp.energy_res_gecco,
            background_model,
            target,
            observation_time,
            sigma,
        )

    @property
    def description(self):
        return f"[dodger_blue1]GECCO ({self._name})"

    @property
    def name(self):
        return f"gecco-{self._name}"

    @staticmethod
    def annihilation_targets():
        """
        Returns a dictionary of the best targets and background models
        for dark matter annihilations using the GECCO telescope.
        """
        return {
            "gc_ein_1_arcmin_cone_optimistic": (
                grp.gc_targets_optimistic["ein"]["1 arcmin cone"],
                gc_bg_model_approx(),
            ),
            "gc_nfw_1_arcmin_cone": (
                grp.gc_targets["nfw"]["1 arcmin cone"],
                gc_bg_model_approx(),
            ),
            "m31_nfw_1_arcmin_cone": (
                grp.m31_targets["nfw"]["1 arcmin cone"],
                grp.gecco_bg_model,
            ),
            "draco_nfw_1_arcmin_cone": (
                grp.draco_targets["nfw"]["1 arcmin cone"],
                grp.gecco_bg_model,
            ),
        }

    @staticmethod
    def decay_targets():
        """
        Returns a dictionary of the best targets and background models
        for dark matter decays using the GECCO telescope.
        """
        return {
            "gc_ein_5_deg_optimistic": (
                grp.gc_targets_optimistic["ein"]["5 deg cone"],
                gc_bg_model_approx(),
            ),
            "gc_nfw_5_deg": (grp.gc_targets["nfw"]["5 deg cone"], gc_bg_model_approx()),
            "m31_nfw_5_deg": (grp.m31_targets["nfw"]["5 deg cone"], grp.gecco_bg_model),
            "draco_nfw_5_deg": (
                grp.draco_targets["nfw"]["5 deg cone"],
                grp.gecco_bg_model,
            ),
        }
