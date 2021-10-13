from cycler import cycler

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
