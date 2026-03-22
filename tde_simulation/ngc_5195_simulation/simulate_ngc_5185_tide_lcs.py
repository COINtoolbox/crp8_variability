import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tidepy
import os

os.environ["TIDE_PATH"]  = "/media3/rupesh/crp8/code/tide_installation/"


def mpc_to_cm(mpc: float) -> float:
    """Convert megaparsecs to centimeters"""
    return mpc * 3.0857e24


def luminosity_to_flux_mjy(luminosity: np.ndarray,
                           d_cm: float,
                           p_nu: float = None) -> np.ndarray:
    """Convert bolometric luminosity to flux density in mJy"""
    if p_nu is not None:
        luminosity = luminosity / p_nu
    flux = luminosity / (4 * np.pi * d_cm**2)
    return flux / 1e-26  

distance_mpc = 9.5
distance_cm = mpc_to_cm(distance_mpc)

c = 2.99792458e10  # cm/s
lambda_f435w = 4350e-8  
nu_f435w = c / lambda_f435w

bh_masses = np.logspace(2, 6, 5)  

for each_mbh in bh_masses:

    p = tidepy.Parameters()
    p.bh_M6 = each_mbh / 1e6
    p.tend = 1000
    p.param_init()
    p.nu = nu_f435w  # F435W nominal frequency

    lc = tidepy.Light_curve_of_tde(p)
    time, luminosity = lc.light_curve()

    flux_mjy = luminosity_to_flux_mjy(luminosity, distance_cm, None)
    df = pd.DataFrame({
        "time_days": time,
        "luminosity": luminosity,
        "flux_mjy": flux_mjy,
        "bh_mass_msun": np.full_like(time, each_mbh)
    })
    df.to_csv(f"tde_lightcurve_bh_{int(each_mbh)}Msun.csv", index=False)
