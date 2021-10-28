from typing import Optional

import numpy as np
import numpy.typing as npt
from rich.progress import Progress
import hazma.single_channel as _single_channel

from . import iterators
from .constrainers import (
    CmbConstrainer,
    GeccoConstrainer,
    EgretConstrainer,
    ComptelConstrainer,
    FermiConstrainer,
    IntegralConstrainer,
    CompositeConstrainer,
)


class SingleChannelConstrainer(CompositeConstrainer):
    """
    Class for computing the constraints on the dark-matter annihilation rate assuming
    the dark-matter annihilates into a single final state.
    """

    def __init__(self, decay):
        super().__init__()
        self.decay = decay
        if decay:
            self._gecco_targets = GeccoConstrainer.decay_targets()
        else:
            self._gecco_targets = GeccoConstrainer.annihilation_targets()
        self.channel = ""

    @property
    def description(self) -> str:
        if self.decay:
            return f"[bold yellow] Single Channel \u03C7 -> {self.channel}"
        else:
            return f"[bold cyan] Single Channel \u03C7\u03C7 -> {self.channel}"

    def gecco_targets(self):
        """
        Return the GECCO targets used.
        """
        return self._gecco_targets

    def set_channel(self, channel):
        if channel == "e e":
            self.channel = "e\u207A + e\u207B"
        elif channel == "mu mu":
            self.channel = "\u03BC\u207A + \u03BC\u207B"
        elif channel == "g g":
            self.channel = "\u03B3 + \u03B3"
        elif channel == "pi pi":
            self.channel = "\u03C0\u207A + \u03C0\u207B"
        else:
            raise ValueError(f"Invalid channel {channel}.")

    def _update(self, model, mx):
        model.mx = mx

    def _model_iterator(self, mxs, finalstate):
        if self.decay:
            base = _single_channel.SingleChannelDec(0.0, finalstate, 1.0)
        else:
            base = _single_channel.SingleChannelAnn(0.0, finalstate, 1.0)
        up = lambda model, mx: self._update(model, mx)
        return iterators.ModelIterator(base).iterate(up, mxs)

    def constrain(
        self,
        mxs: npt.NDArray[np.float64],
        finalstate: str,
        cmb: bool = True,
        existing: bool = True,
        gecco: bool = True,
        progress: Optional[Progress] = None,
        **options,
    ):
        """
        Compute the constraints on simple annihilating DM models where the DM
        annihilates only into a specified final state. The observations used
        to constrain are: CMB, COMPTEL, EGRET, Fermi, INTEGRAL and GECCO.

        Parameters
        ----------
        mxs: ArrayLike
            Array of dark-matter masses.
        finalstate: str
            Final state the dark-matter decay/annihilates into.
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
        channels: List[str]
            Channels to constrain.

        Returns
        -------
        constraints: Dict
            Dictionary containing all constraints.
        """
        self.reset_constrainers()
        model_iterator = self._model_iterator(mxs, finalstate)
        self.set_channel(finalstate)

        if cmb and not self.decay:
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

        return self._constrain(model_iterator, progress)


def example_config():
    from .utils import print_example_config

    required = {
        "decay": "true",
        "filename": "'single_channel_dec'",
    }
    optional = {
        "mx-min": 0.1,
        "mx-max": 500.0,
        "num-mx": 100,
        "channels": ["e e", "mu mu", "g g"],
        "sigma": [5.0, 25.0],
        "overwrite": "false",
    }
    print_example_config(required, optional)


if __name__ == "__main__":
    from .utils import parse_json, make_parser, write_data
    from hazma.parameters import (
        electron_mass as me,
        muon_mass as mmu,
        charged_pion_mass as mpi,
    )

    parser = make_parser(
        "single-channel", "Compute constraints on the single-channel model."
    )

    config = parse_json(parser.parse_args(), ["decay"], example_config)
    decay = config["decay"]

    sigma = config["sigma"]
    max_mx = config["max-mx"]
    min_mx = config["min-mx"]
    num_mx = config["num-mx"]
    channels = config.get("channels", ["e e", "mu mu", "g g"])

    masses = {}
    if "e e" in channels:
        masses["e e"] = np.geomspace(max(me, min_mx), max_mx, num_mx)
    if "mu mu" in channels:
        masses["mu mu"] = np.geomspace(max(mmu, min_mx), max_mx, num_mx)
    if "g g" in channels:
        masses["g g"] = np.geomspace(min_mx, max_mx, num_mx)
    if "pi pi" in channels:
        masses["pi pi"] = np.geomspace(max(mpi, min_mx), max_mx, num_mx)

    cmb = not decay

    sc = SingleChannelConstrainer(decay)

    constraints = {}
    for fs, mxs in masses.items():
        with Progress() as progress:
            constraints[fs] = sc.constrain(
                mxs,
                fs,
                decay=decay,
                progress=progress,
                sigma=sigma,
                cmb=cmb,
                channels=channels,
            )
            constraints[fs]["masses"] = mxs

    write_data(config["prefix"], config["filename"], constraints, config["overwrite"])
