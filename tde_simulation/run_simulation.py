"""Simulate TDEs with TiDE package"""

import argparse
import glob
import json
import os
from io import StringIO

import requests
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import tidepy

from utils import (
    mag_to_flux,
    preprocess_light_curve,
    sample_tide_configs,
    simulate_flux_with_tide,
    plot_light_curves_combined,
    load_config,
    redshift_to_cm,
    mpc_to_cm,
    luminosity_to_flux_mjy,
    fetch_ztf_data_by_coordinates
)

ZTF_NOMINAL_FREQUENCY_MAPPING = {
    "zg": 6.35e14,
    "zr": 4.73e14,
    "zi": 3.8e14
}


def generate_tide_data(config: Dict[str, Any],
                       tide_config: Dict[str, Any],
                       band_filter: str) -> pd.Series:
    """Generates TiDE simulation data for a given configuration
    Args:
        config: Configuration dictionary.
        tide_config: Tide configuration dictionary.
        band_filter: The filter to use for simulation.

    Returns:
        A pandas Series containing the simulated flux.

    Raises:
        ValueError: If both 'distance_mpc' and 'redshift' are not specified.
    """
    distance_mpc: Optional[float] = config.get("distance_mpc")
    redshift: Optional[float] = config.get("redshift")

    if (distance_mpc is None and redshift is None) or \
       (distance_mpc is not None and redshift is not None):
        raise ValueError("Specify either 'distance_mpc'"
                         " or 'redshift', not both.")

    p = tidepy.Parameters()
    p.bh_M6 = tide_config["bh_mass"] / 1e6
    p.star_mstar = tide_config["star_mstar"]
    p.star_rstar = tide_config["star_rstar"]
    p.tend = 1000
    p.dt = 1
    p.nu = ZTF_NOMINAL_FREQUENCY_MAPPING[band_filter]
    p.param_init()
    lc = tidepy.Light_curve_of_tde(p)
    results = lc.light_curve()
    luminosity: np.ndarray = results[1]

    if config["use_p_nu_scaling"]:
        luminosity *= p.nu

    if redshift is not None:
        distance_cm = redshift_to_cm(redshift)
    else:
        distance_cm = mpc_to_cm(distance_mpc)

    flux_mjy = luminosity_to_flux_mjy(
        luminosity, distance_cm, p.nu if tide_config["use_pnu"] else None)
    return pd.Series(flux_mjy)


def simulate_data(ztf_data: pd.DataFrame, config: Dict[str, Any],
                  tide_config: Dict[str, Any],
                  band_filter: str) -> pd.DataFrame:
    """Simulates TDE data by generating tidal flux and combining with ZTF data.

    """
    tide_flux_mjy = generate_tide_data(config, tide_config, band_filter)

    ztf_data["flux"] = mag_to_flux(ztf_data["mag"])
    ztf_data = simulate_flux_with_tide(ztf_data,
                                       tide_flux_mjy, tide_config["start_mjd"])

    return ztf_data


def create_output_dirs(config: Dict[str, Any]) -> None:
    """Creates the necessary output directories.

    Args:
        config: Configuration dictionary.
    """
    os.makedirs(os.path.join(config["output_dir"], "data"), exist_ok=True)
    os.makedirs(os.path.join(config["output_dir"], "config"), exist_ok=True)
    if config["save_plots"]:
        os.makedirs(os.path.join(config["output_dir"], "plots"), exist_ok=True)


def save_output(combined_df: pd.DataFrame, config: Dict[str, Any],
                tide_config: Dict[str, Any], file_name: str):
    output_dir = config["output_dir"]
    combined_df.to_csv(
        os.path.join(output_dir, "data", file_name + ".csv")
    )

    if config["save_plots"]:
        plot_light_curves_combined(combined_df, config, tide_config)

    with open(
            os.path.join(output_dir, "config", file_name + ".json"),
            "w",
            encoding="utf-8",
    ) as f:
        json.dump(tide_config, f, indent=4)


def process_single_object(ztf_data: pd.DataFrame, object_id: str, config: Dict[str, Any]) -> None:
    ztf_data["flux"] = mag_to_flux(ztf_data["mag"])
    ztf_data = preprocess_light_curve(ztf_data)

    tide_config = sample_tide_configs(config)
    start_mjd = np.random.choice(ztf_data["mjd"].values)
    tide_config.update({"start_mjd": start_mjd})
    config["file_name"] = object_id

    all_band_data = []
    for band_filter in config.get("filters", []):
        band_data = ztf_data[ztf_data["filtercode"] == band_filter].copy()
        if band_data.empty:
            continue

        band_data = band_data.sort_values("mjd").reset_index(drop=True)
        simulated_data = simulate_data(band_data, config, tide_config, band_filter)

        if not simulated_data.empty:
            all_band_data.append(simulated_data)

    if all_band_data:
        output_df = pd.concat(all_band_data)
        save_output(output_df, config, tide_config, object_id)


def main(config_path: str) -> None:
    config = load_config(config_path)

    tide_path = config.get("tide_path")
    if not tide_path:
        raise ValueError("Missing 'tide_path' in config")

    os.environ["TIDE_PATH"] = tide_path
    print(f"[INFO] TIDE_PATH set to: {tide_path}")
    create_output_dirs(config)

    if "coordinates_file" in config:
        coordinates_df = pd.read_csv(config["coordinates_file"])
        search_radius_deg = config["search_radius_arcsec"] / 3600

        for _, row in coordinates_df.iterrows():
            obj_id = str(row["objectId"])
            try:
                raw_data = fetch_ztf_data_by_coordinates(row["ra"], row["dec"], search_radius_deg)
                ztf_df = pd.read_csv(StringIO(raw_data))
                process_single_object(ztf_df, obj_id, config)
            except Exception as e:
                print(f"[ERROR] Failed to fetch/process data for {obj_id}: {e}")

    elif "ztf_files_path" in config:
        for file_path in glob.glob(config["ztf_files_path"]):
            obj_id = os.path.splitext(os.path.basename(file_path))[0]
            ztf_df = pd.read_csv(file_path)
            process_single_object(ztf_df, obj_id, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TDE light curve simulation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration file",
    )

    args = parser.parse_args()
    main(args.config)
