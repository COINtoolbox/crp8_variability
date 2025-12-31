"""
 Utils file
"""

import json
import os
import random
from typing import Optional, Dict, Any, Union

from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def redshift_to_cm(z: float) -> float:
    """Convert redshift to comoving distance in centimeters"""
    return cosmo.comoving_distance(z).to(u.cm).value


def mag_to_flux(mag):
    """Convert AB Magnitude to flux in milliJansky"""
    return np.power(10, (26 - (mag + 48.6) / 2.5))


def mpc_to_cm(mpc: float) -> float:
    """Convert megaparsecs to centimeters"""
    return mpc * 3.0857e24


def luminosity_to_flux_mjy(luminosity: np.ndarray,
                           d_cm: float,
                           p_nu: Optional[float] = None) -> np.ndarray:
    """Convert luminosity to flux in millijanskys"""
    flux = luminosity / (4 * np.pi * d_cm**2)
    if p_nu is not None:
        flux /= p_nu
    return flux / 1e-26


def flux_to_mag(flux):
    """Convert flux from milliJansky to AB Magnitude."""
    f_cgs = flux * 1e-3 * 1e-23  # mJy → Jy → erg/s/cm²/Hz
    mag = -2.5 * np.log10(f_cgs) - 48.6
    return mag


def load_config(path: str) -> Dict[str, Any]:
    """Load a JSON configuration file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_light_curve(df):
    """Sort and reset index for a single OID+Filter light curve"""
    return df.sort_values(by="mjd").reset_index(drop=True)


def sample_star_rstar(cfg: Dict[str, Any]) -> Union[float, int]:
    """Sample star radius based on configuration"""
    mode = cfg["mode"]
    if mode == "categorical":
        return random.choice(cfg["categorical"])
    elif mode == "continuous":
        return round(np.random.uniform(cfg["continuous"]["min"],
                                      cfg["continuous"]["max"]), 2)
    elif mode == "mixed":
        if random.random() < 0.5:
            return random.choice(cfg["categorical"])
        else:
            return round(np.random.uniform(cfg["continuous"]["min"],
                                          cfg["continuous"]["max"]), 2)
    else:
        raise ValueError(f"Invalid star_rstar mode: {mode}")


def sample_tide_configs(config):
    """Samples TiDE configs from the parameter space defined in config"""
    bh_mass: float = np.random.uniform(config["bh_mass"]["min"],
                                       config["bh_mass"]["max"])
    star_mstar: float = np.random.uniform(config["star_mstar"]["min"],
                                         config["star_mstar"]["max"])
    star_rstar = sample_star_rstar(config["star_rstar"])
    use_pnu: bool = random.choice(config["use_p_nu_scaling"])
    return {"bh_mass": bh_mass,
            "star_mstar": star_mstar,
            "star_rstar": star_rstar,
            "use_pnu": use_pnu}


def plot_light_curves_combined(df, config, tide_config):
    """Plot original vs simulated ZTF mag in separate subplots"""
    _, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    output_file = os.path.join(config["output_dir"],
                               "plots", config["file_name"])
    colors = {"zg": "#141E3C", "zr": "#FF5B0B", "zi": "orange"}

    for fid in df["filtercode"].unique():
        f_data = df[df["filtercode"] == fid]
        color = colors.get(fid, "black")
        axes[0].scatter(f_data["mjd"], f_data["original_mag"],
                        color=color, label=f"Original {fid}",
                        marker="o", alpha=0.3, s=40)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("AB Magnitude", fontsize=12)
    axes[0].set_title(f"Original Light Curve\n Object ID: "
                      f"{config['file_name']}", fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    for fid in df["filtercode"].unique():
        f_data = df[df["filtercode"] == fid]
        color = colors.get(fid, "black")
        axes[1].scatter(f_data["mjd"], f_data["simulated_mag"],
                        color=color, label=f"Simulated {fid}",
                        marker="s", alpha=0.6, s=40, linewidth=0.5)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("MJD)", fontsize=12)
    axes[1].set_ylabel("AB Magnitude", fontsize=12)

    bh_mass = tide_config.get("bh_mass")
    star_mstar = tide_config.get("star_mstar")

    bh_mass_str = (
        f"{bh_mass / 1e6:.2f}" if isinstance(bh_mass, (int, float)) else "N/A"
    )
    star_mstar_str = (
        f"{star_mstar:.2f}" if isinstance(star_mstar, (int, float)) else "N/A"
    )

    star_rstar = tide_config.get("star_rstar", "N/A")
    start_mjd = int(tide_config.get("start_mjd", "N/A"))

    if "distance_mpc" in config:
        distance_info = f"Distance: {config['distance_mpc']} Mpc"
    elif "redshift" in config:
        distance_info = f"Redshift: {config['redshift']}"
    else:
        distance_info = ""

    tde_info = (
        f"BH Mass: {bh_mass_str} ×10⁶ M☉, "
        f"Star Mass: {star_mstar_str} M☉, "
        f"Star Radius: {star_rstar} R☉, "
        f"{distance_info}, "
        f"Start MJD: {start_mjd}"
    )

    axes[1].set_title(f"Simulated Light Curve\n"
                      f"{tde_info}", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_file)


def simulate_flux_with_tide(real_df, tide_df, start_mjd):
    """Simulates light curve by injecting synthetic
     flux into observed light curve"""
    if len(real_df) < 2:
        print("Not enough data points.")
        return pd.DataFrame([])

    time_diffs = np.abs(real_df["mjd"] - start_mjd)
    start_index = time_diffs.idxmin()
    real_df = real_df.copy()
    real_df["timedays"] = real_df["mjd"]
    real_df.loc[start_index:, "timedays"] = (
            real_df["mjd"].iloc[start_index:] -
            real_df["mjd"].iloc[start_index:].min()).round().astype(int)

    x1 = real_df["timedays"].iloc[start_index:-1].reset_index(drop=True)
    x2 = real_df["timedays"].iloc[start_index + 1:].reset_index(drop=True)
    valid = (x1 < len(tide_df)) & (x2 < len(tide_df))

    x1 = x1[valid]
    y1 = tide_df.iloc[x1.values].reset_index(drop=True)
    flux_diff = y1.to_numpy()

    updated_flux = real_df["flux"].copy()
    updated_flux.iloc[start_index + 1:start_index + 1 + len(
        flux_diff)] += flux_diff

    real_df["simulated_flux"] = updated_flux
    real_df["simulated_mag"] = flux_to_mag(real_df["simulated_flux"])
    real_df["original_mag"] = flux_to_mag(real_df["flux"])
    return real_df
