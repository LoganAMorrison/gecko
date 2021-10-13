import hazma.gamma_ray_parameters as grp


def gc_bg_model_approx():
    """
    Returns a background model simple to `hazma.gamma_ray_parameters.gc_bg_model`
    but it does not constrain the input energies.
    """
    return grp.BackgroundModel(
        [0, 1e5], lambda e: 7 * grp.default_bg_model.dPhi_dEdOmega(e)
    )


def effective_area():
    """
    Returns the effective area for the GECCO telescope.
    """
    return grp.effective_area_gecco


def energy_resolution():
    """
    Returns the energy resulution for the GECCO telescope.
    """
    return grp.energy_res_gecco


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
        "draco_nfw_5_deg": (grp.draco_targets["nfw"]["5 deg cone"], grp.gecco_bg_model),
    }
