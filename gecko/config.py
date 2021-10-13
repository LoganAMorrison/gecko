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
    "gc_nfw_1_arcmin_cone": r"(GC $1'$, NFW)",
    "gc_nfw_5_deg": r"(GC $5^\circ$, NFW)",
    "gc_ein_1_arcmin_cone_optimistic": r"(GC $1'$, Einasto)",
    "gc_ein_5_deg_optimistic": r"(GC $5^\circ$, Einasto)",
    "m31_nfw_1_arcmin_cone": r"(M31 $1'$)",
    "m31_nfw_5_deg": r"(M31 $5^\circ$)",
    "draco_nfw_1_arcmin_cone": r"(Draco $1'$)",
    "draco_nfw_5_deg": r"(Draco $5^\circ$)",
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
    "gc_nfw_1_arcmin_cone": BROWN,
    "gc_ein_1_arcmin_cone_optimistic": PURPLE,
    "m31_nfw_1_arcmin_cone": PINK,
    "draco_nfw_1_arcmin_cone": GREY,
    "cmb_s_wave": BLACK,
    "cmb_p_wave": BLACK,
    "gc_nfw_5_deg": BROWN,
    "gc_ein_5_deg_optimistic": PURPLE,
    "m31_nfw_5_deg": PINK,
    "draco_nfw_5_deg": GREY,
}
