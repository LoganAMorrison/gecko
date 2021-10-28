from cycler import cycler
from typing import Optional, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np

from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
TAU_TEX = r"$\tau \ [\mathrm{s}]$"
RHN_MASS_TEX = r"$m_{N} \ [\mathrm{MeV}]$"
DM_MASS_TEX = r"$m_{\chi} \ [\mathrm{MeV}]$"


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


def get_gecco_keys(datafile):
    return list(filter(lambda k: "gecco" in k, list(datafile.keys())))


def strip_gecco(key: str):
    return key.replace("gecco-", "")


def add_gecco(axis, masses, datafile, decay=False, upper_band=5, lower_band=25):
    for key in get_gecco_keys(datafile):
        conf = {"color": COLOR_DICT[strip_gecco(key)]}
        c1 = (upper_band / 5) * datafile[key][:]
        c2 = (lower_band / 5) * datafile[key][:]
        if decay:
            c1 = 1 / c1
            c2 = 1 / c2

        avg = np.exp(np.log(c1 * c2) / 2.0)
        axis.fill_between(masses, c1, c2, lw=0, alpha=0.4, **conf)
        axis.plot(masses, avg, lw=1.5, **conf)
        axis.plot(masses, c1, lw=0.0, alpha=0.3, ls="-", **conf)
        axis.plot(masses, c2, lw=0.0, alpha=0.3, ls="-", **conf)


def existing_outline(datafile, decay=False):
    egret = datafile["egret"][:]
    comptel = datafile["comptel"][:]
    fermi = datafile["fermi"][:]
    integral = datafile["integral"][:]
    outline = np.array(
        [np.min([e, c, f, i]) for e, c, f, i in zip(egret, comptel, fermi, integral)]
    )
    if decay:
        return 1 / outline
    return outline


def add_existing_outline(axis, masses, ylims, datafile, decay=False, **options):
    outline = existing_outline(datafile, decay)
    outline = np.clip(outline, np.min(ylims), np.max(ylims))

    if decay:
        y2 = np.min(ylims)
    else:
        y2 = np.max(ylims)

    alpha = options.get("alpha", 0.5)
    axis.fill_between(masses, outline, y2, alpha=alpha)
    axis.plot(masses, outline)


def configure_ticks(axis, minor_ticks="both"):
    axis.tick_params(axis="both", which="both", direction="in", width=0.9, labelsize=12)
    if minor_ticks in ["y", "both"]:
        axis.yaxis.set_minor_locator(
            LogLocator(base=10, subs=[i * 0.1 for i in range(1, 10)], numticks=100)
        )
        axis.yaxis.set_minor_formatter(NullFormatter())
    if minor_ticks in ["x", "both"]:
        axis.xaxis.set_minor_locator(
            LogLocator(base=10, subs=[i * 0.1 for i in range(1, 10)], numticks=100)
        )
        axis.xaxis.set_minor_formatter(NullFormatter())


def add_xy_grid(axis, alpha=0.5):
    axis.grid(True, axis="y", which="major", alpha=alpha)
    axis.grid(True, axis="x", which="major", alpha=alpha)


CMB_HANDLE = Line2D([0], [0], color="k", label=LABEL_DICT["cmb"], ls="--", lw=1)
RD_HANDLE = Line2D([0], [0], color="k", label=LABEL_DICT["rd"], ls="dotted", lw=1)
PHENO_HANDLE = Patch(color=COLOR_DICT["pheno"], label=LABEL_DICT["pheno"], alpha=0.3)
EXISTING_HANDLE = Patch(
    color=COLOR_DICT["existing"], label=LABEL_DICT["existing"], alpha=0.3
)


def gecco_handle(name):
    return Line2D(
        [0],
        [0],
        color=COLOR_DICT[name],
        label="GECCO" + LABEL_DICT[name],
        alpha=1,
    )


def get_legend_handles(datafile, **kwargs):
    handles = []
    if kwargs.get("cmb", False):
        handles.append(CMB_HANDLE)
    if kwargs.get("rd", False):
        handles.append(RD_HANDLE)
    for key in get_gecco_keys(datafile):
        handles.append(gecco_handle(strip_gecco(key)))
    if kwargs.get("pheno", False):
        handles.append(PHENO_HANDLE)
    if kwargs.get("existing", False):
        handles.append(EXISTING_HANDLE)
    return handles


def make_legend_axis(axis, datafile, **kwargs):
    handles = get_legend_handles(datafile, **kwargs)

    axis.clear()
    axis.set_axis_off()
    axis.legend(handles=handles, loc="center", fontsize=10)


def add_legend(axis, datafile, **kwargs):
    handles = get_legend_handles(datafile, **kwargs)

    axis.legend(handles=handles, **kwargs)


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
