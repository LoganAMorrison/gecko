import warnings
from copy import copy
from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import root_scalar, RootResults
from typing import Optional, Tuple
from rich.progress import Progress, TaskID

from hazma.relic_density import relic_density
from hazma.parameters import omega_h2_cdm

from .utils import sigmav


class AbstractConstrainer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _constrain(self, model):
        pass

    def constrain(
        self, model_iterator, progress_task: Optional[Tuple[Progress, TaskID]] = None
    ):
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
        if progress_task is None:
            progress_update = lambda: None
        else:
            progress_update = lambda: progress_task[0].update(
                progress_task[1], advance=1, refresh=True
            )

        constraints = np.zeros((len(model_iterator),), dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, model in enumerate(model_iterator):
                constraints[i] = self._constrain(model)
                progress_update()
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


class CmbConstrainer(AbstractConstrainer):
    """
    Class for computing constraints on dark-matter models from CMB.
    """

    def __init__(self, x_kd=1e-6):
        super().__init__()
        self.x_kd = x_kd

    def _constrain(self, model):
        return model.cmb_limit(x_kd=self.x_kd)


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
