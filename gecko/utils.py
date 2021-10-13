from typing import Union, Dict, Optional

from hazma.scalar_mediator import HiggsPortal
from hazma.vector_mediator import KineticMixing
from hazma.parameters import sv_inv_MeV_to_cm3_per_s


def sigmav(model: Union[HiggsPortal, KineticMixing], vx: float = 1e-3):
    """Compute <σv> for the given model."""
    cme = 2 * model.mx * (1.0 + 0.5 * vx ** 2)
    sig = model.annihilation_cross_sections(cme)["total"]
    return sig * vx * sv_inv_MeV_to_cm3_per_s


def add_to_dataset(dataset, dictionary: Dict, path: Optional[str] = None):
    """Add a nested dictionary to the hdf5 dataset."""
    for k, v in dictionary.items():
        if path is not None:
            newpath = "/".join([path, k])
        else:
            newpath = f"{k}"
        if isinstance(v, dict):
            add_to_dataset(dataset, v, newpath)
        else:
            # at the bottom
            # print(f"adding {newpath}")
            dataset.create_dataset(newpath, data=v)
