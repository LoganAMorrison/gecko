from cycler import cycler
from typing import List, Optional, Tuple, Union
from abc import ABC, abstractmethod

LineStyleType = Union[str, Tuple[int, Tuple[int, ...]]]

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
PURPLE = "#9467bd"
BROWN = "#8c564b"
PINK = "#e377c2"
GREY = "#7f7f7f"
YELLOW_GREEN = "#bcbd22"
TEAL = "#17becf"
BLACK = "k"

BR_TEX = r"$\mathrm{Br}$"
SIGV_TEX = r"${\langle\sigma v\rangle}_{\bar{\chi}\chi,0}$"
SIGV_UNITS = r"$[\mathrm{cm}^3/\mathrm{s}]$"

MEV_UNITS = r"$[\mathrm{MeV}]$"


GECCO_ANNS = [
    "draco_nfw_1_arcmin_cone",
    "gc_ein_1_arcmin_cone_optimistic",
    "gc_nfw_1_arcmin_cone",
    "m31_nfw_1_arcmin_cone",
]
GECCO_DECS = [
    "draco_nfw_5_deg",
    "gc_ein_5_deg_optimistic",
    "gc_nfw_5_deg",
    "m31_nfw_5_deg",
]
EXISTINGS = ["integral", "comptel", "egret", "fermi"]
CMBS = ["cmb_p_wave", "cmb_s_wave"]


LABEL_DICT = {
    "e e": r"$e^+ e^-$",
    "mu mu": r"$\mu^+ \mu^-$",
    "pi pi": r"$\pi^+ \pi^-$",
    "pi0 pi0": r"$\pi^0 \pi^0$",
    "pi0 g": r"$\pi^0 \gamma$",
    "g g": r"$\gamma \gamma$",
    "comptel": r"COMPTEL",
    "egret": r"EGRET",
    "fermi": r"Fermi",
    "integral": r"INTEGRAL",
    "pheno": r"Other Constraints",
    "existing": r"$\gamma$-ray telescopes",
    "rd": r"Thermal relic",
    "gc_nfw_1_arcmin_cone": r"(GC $1'$, NFW)",
    "gc_nfw_5_deg": r"(GC $5^\circ$, NFW)",
    "gc_ein_1_arcmin_cone_optimistic": r"(GC $1'$, Einasto)",
    "gc_ein_5_deg_optimistic": r"(GC $5^\circ$, Einasto)",
    "m31_nfw_1_arcmin_cone": r"(M31 $1'$)",
    "m31_nfw_5_deg": r"(M31 $5^\circ$)",
    "draco_nfw_1_arcmin_cone": r"(Draco $1'$)",
    "draco_nfw_5_deg": r"(Draco $5^\circ$)",
    "cmb": r"CMB",
    "cmb_s_wave": r"CMB ($s$-wave)",
    "cmb_p_wave": r"CMB ($p$-wave)",
}


MPL_LINESTYLES = {
    "solid": (0, (1, 0)),
    "dashdot": (0, (5, 1, 1, 1)),
    "loosely dotted": (0, (1, 10)),
    "loosely dotted": (0, (1, 10)),
    "dotted": (0, (1, 1)),
    "densely dotted": (0, (1, 1)),
    "loosely dashed": (0, (5, 10)),
    "dashed": (0, (5, 5)),
    "densely dashed": (0, (5, 1)),
    "loosely dashdotted": (0, (3, 10, 1, 10)),
    "dashdotted": (0, (3, 5, 1, 5)),
    "densely dashdotted": (0, (3, 1, 1, 1)),
    "dashdotdotted": (0, (3, 5, 1, 5, 1, 5)),
    "loosely dashdotdotted": (0, (3, 10, 1, 10, 1, 10)),
    "densely dashdotdotted": (0, (3, 1, 1, 1, 1, 1)),
}

COLOR_DICT = {
    "comptel": BLUE,
    "egret": ORANGE,
    "fermi": GREEN,
    "integral": RED,
    "pheno": ORANGE,
    "existing": BLUE,
    "gc_nfw_1_arcmin_cone": BROWN,
    "gc_ein_1_arcmin_cone_optimistic": PURPLE,
    "m31_nfw_1_arcmin_cone": PINK,
    "draco_nfw_1_arcmin_cone": GREY,
    "cmb": BLACK,
    "cmb_s_wave": BLACK,
    "cmb_p_wave": BLACK,
    "gc_nfw_5_deg": BROWN,
    "gc_ein_5_deg_optimistic": PURPLE,
    "m31_nfw_5_deg": PINK,
    "draco_nfw_5_deg": GREY,
}

MPL_COLORS = cycler(
    color=[
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
)

MPL_LINESTYLES = {
    "solid": (0, (1, 0)),
    "dashdot": (0, (5, 1, 1, 1)),
    "loosely dotted": (0, (1, 10)),
    "loosely dotted": (0, (1, 10)),
    "dotted": (0, (1, 1)),
    "densely dotted": (0, (1, 1)),
    "loosely dashed": (0, (5, 10)),
    "dashed": (0, (5, 5)),
    "densely dashed": (0, (5, 1)),
    "loosely dashdotted": (0, (3, 10, 1, 10)),
    "dashdotted": (0, (3, 5, 1, 5)),
    "densely dashdotted": (0, (3, 1, 1, 1)),
    "dashdotdotted": (0, (3, 5, 1, 5, 1, 5)),
    "loosely dashdotdotted": (0, (3, 10, 1, 10, 1, 10)),
    "densely dashdotdotted": (0, (3, 1, 1, 1, 1, 1)),
}


class MatPlotLibConfig:
    DEFAULT_DICT = {
        "label": None,
        "color": "k",
        "linestyle": MPL_LINESTYLES["solid"],
        "alpha": 0.7,
        "linewidth": 0.5,
    }

    def __init__(self):
        self._dict = {}

    def add_key(self, key, **kwargs):
        if not (key in self._dict.keys()):
            self._dict[key] = self.DEFAULT_DICT
        for k, v in kwargs.items():
            self._dict[key][k] = v

    def __call__(self, key):
        if not (key in self._dict.keys()):
            self._dict[key] = self.DEFAULT_DICT
        return self._dict[key]


class Label:
    def __init__(
        self, text, fontsize: Optional[int] = None, units: Optional[str] = None
    ):
        self.text = text
        self.fontsize = fontsize
        self.units = units


class Limits:
    def __init__(self, low, high):
        self.low = low
        self.high = high


class PlotItem(ABC):
    def __init__(
        self,
        linewidth: Optional[int] = None,
        linestyle: Optional[LineStyleType] = None,
        color: Optional[str] = None,
        alpha: Optional[int] = None,
        label: Optional[Label] = None,
    ):
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.color = color
        self.alpha = alpha
        self.label = label

    def config(self):
        conf = {}
        if self.linewidth is not None:
            conf["linewidth"] = self.linewidth
        if self.linestyle is not None:
            conf["linestyle"] = self.linestyle
        if self.color is not None:
            conf["color"] = self.color
        if self.alpha is not None:
            conf["alpha"] = self.alpha
        if self.label is not None:
            conf["label"] = self.label.text
        return conf

    @abstractmethod
    def add_to_axis(self, axis):
        pass


class Line(PlotItem):
    def __init__(self, x, y, **options):
        super().__init__(**options)
        self.x = x
        self.y = y

    def add_to_axis(self, axis):
        axis.plot(self.x, self.y, **self.config())


class Band(PlotItem):
    def __init__(self, x, y1, y2, **options):
        super().__init__(**options)
        self.x = x
        self.y1 = y1
        self.y2 = y2

    def add_to_axis(self, axis):
        axis.fill_between(self.x, self.y1, self.y2, **self.config())


class BandedLine(PlotItem):
    def __init__(self, lower: Line, upper: Line, center: Line, band: Band):
        self.lower = lower
        self.upper = upper
        self.center = center
        self.band = band

    def add_to_axis(self, axis):
        self.band.add_to_axis(axis)
        self.lower.add_to_axis(axis)
        self.upper.add_to_axis(axis)
        self.center.add_to_axis(axis)
