
import os
import glob
from multiprocessing import Pool
from typing import Dict, Tuple, Optional, List, Union


import numpy as np
import pandas as pd
from tqdm import tqdm


from utils import (
    load_snad_data,
    mag_to_flux,
    distance_suffix,
    shift_magnitude_to_distance,
    compute_reference_mJy,
    dc_to_difference_mag,
    compute_snr,
    is_trigger_alert,
    plot_combined_mags_per_distance,
    simulate_flux_with_tide, distance_to_z,
    redraw_at_distance,
)


DR_FILES_LIST = glob.glob("/media3/rupesh/crp8/data/legus_photometry/snad/*.json")
BAND_MAPPING = {"zg": 1, "zr": 2}
SIMULATION_BAND_MAPPING = {"zg": "flux_mjy_g", "zr": "flux_mjy_r"}
OUTPUT_DIR = "/media3/rupesh/crp8/data/legus_photometry/snad_simulation_fixed_final_100mpc"
DISTANCES_TO_TRANSFORM = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

ORIGINAL_DISTANCE = 9.5
SAVE_PLOTS = True


def _compute_distance_variant(
    injected_flux: np.ndarray,
    flux_err: np.ndarray,
    d_new: float,
    reference_mag: float,
    reference_mag_err: float,
    mjd: Union[pd.Series, np.ndarray],
    start_mjd: float,
) -> Dict[str, np.ndarray]:

    mag, err = redraw_at_distance(injected_flux, flux_err, ORIGINAL_DISTANCE, d_new)

    reference_mag_shifted = reference_mag
    reference_mag_err_scaled = reference_mag_err
    if d_new != ORIGINAL_DISTANCE:
        reference_mag_shifted = shift_magnitude_to_distance(
            reference_mag,
            ORIGINAL_DISTANCE,
            d_new
        )
        reference_mag_err_scaled = reference_mag_err * (d_new / ORIGINAL_DISTANCE) ** 2

    if d_new != ORIGINAL_DISTANCE:
        z_old = distance_to_z(ORIGINAL_DISTANCE)
        z_new = distance_to_z(d_new)

        mjd = np.asarray(mjd, dtype=float)

        t_rest = mjd - start_mjd

        scale = (1 + z_new) / (1 + z_old)
        mjd = t_rest * scale + start_mjd

    diff_data = np.array([
        dc_to_difference_mag(m, me, reference_mag_shifted, reference_mag_err_scaled)
        for m, me in zip(mag, err)
    ])

    magpsf, sigmapsf, idp = diff_data.T
    snr = compute_snr(sigmapsf)

    return {
        "simulated_mag": mag,
        "simulated_mag_error": err,
        "simulated_mjd": mjd,
        "is_dimmer": mag > reference_mag_shifted,
        "magpsf": magpsf,
        "sigmapsf": sigmapsf,
        "idp": idp,
        "snr": snr,
        "is_alert": is_trigger_alert(snr, idp, mjd, start_mjd),
        "reference_mag": reference_mag_shifted,
    }

def process_distance_block(
    sim_base: pd.DataFrame,
    reference_mag: float,
    reference_mag_err: float,
    start_mjd: float,
) -> pd.DataFrame:
    results: pd.DataFrame = sim_base[['mjd', "mag", "magerr"]].copy()

    injected_flux: np.ndarray = sim_base["injected_flux"].to_numpy(dtype=float)
    flux_err: np.ndarray = sim_base["flux_err"].to_numpy(dtype=float)
    mjd: pd.Series = results["mjd"]

    for d in [ORIGINAL_DISTANCE, *DISTANCES_TO_TRANSFORM]:
        suffix: str = distance_suffix(d)

        computed = _compute_distance_variant(
            injected_flux, flux_err, d,
            reference_mag, reference_mag_err,
            mjd, start_mjd
        )

        for key, value in computed.items():
            results[f"{key}{suffix}"] = value

    return results


def process_filter(
    dr_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    dr_filter: str,
    start_mjd: float,
) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    filter_df: pd.DataFrame = (
        dr_df.loc[dr_df["filtercode"] == dr_filter]
        .sort_values("mjd")
        .reset_index(drop=True)
    )

    if filter_df.empty:
        return None, None

    _, _, ref_mag, ref_mag_err = compute_reference_mJy(filter_df)

    sim_filter: pd.DataFrame = sim_df[
        ["time_days", SIMULATION_BAND_MAPPING[dr_filter]]
    ].reset_index(drop=True)

    sim_base: pd.DataFrame = simulate_flux_with_tide(
        filter_df,
        sim_filter,
        start_mjd,
        SIMULATION_BAND_MAPPING[dr_filter],
    )

    if sim_base.empty:
        return None, None

    result: pd.DataFrame = process_distance_block(
        sim_base, ref_mag, ref_mag_err, start_mjd
    )
    result["filter"] = dr_filter

    return result, ref_mag


def process_single_dr_file(
    sim_df: pd.DataFrame,
    dr_file: str,
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[Dict[str, float]],
    Optional[float]
]:
    dr_df: pd.DataFrame = load_snad_data(dr_file)
    if dr_df.empty:
        return None, None, None

    dr_df["flux"] = mag_to_flux(dr_df["mag"])

    start_mjd = np.random.uniform(dr_df["mjd"].min(), dr_df["mjd"].max())

    results: List[pd.DataFrame] = []
    ref_mags: Dict[str, float] = {}

    for dr_filter in BAND_MAPPING:
        res, ref_mag = process_filter(dr_df, sim_df, dr_filter, start_mjd)
        if res is None or ref_mag is None:
            continue

        results.append(res)
        ref_mags[dr_filter] = ref_mag

    if not results:
        return None, None, None

    return pd.concat(results, ignore_index=True), ref_mags, start_mjd


def run_pipeline(sim_file: str) -> None:
    print("Processing:", sim_file)

    sim_fname: str = os.path.splitext(os.path.basename(sim_file))[0]
    output_dir: str = os.path.join(OUTPUT_DIR, sim_fname)
    os.makedirs(output_dir, exist_ok=True)

    sim_df: pd.DataFrame = pd.read_csv(sim_file)
    bh_mass: float = float(sim_df["bh_mass_msun"].iloc[0])

    for dr_file in tqdm(DR_FILES_LIST):
        object_id: str = os.path.splitext(os.path.basename(dr_file))[0]

        sim_results, ref_mag_dict, start_mjd = process_single_dr_file(sim_df, dr_file)
        if sim_results is None or ref_mag_dict is None or start_mjd is None:
            continue

        sim_results["start_mjd"] = start_mjd
        sim_results["object_id"] = object_id
        sim_results["bh_mass_msun"] = bh_mass

        for f, ref_mag in ref_mag_dict.items():
            sim_results[f"reference_mag_{f}"] = ref_mag

        output_path: str = os.path.join(output_dir, f"{object_id}.csv")
        sim_results.to_csv(output_path, index=False)
        if SAVE_PLOTS:
            plot_combined_mags_per_distance(
                sim_results,
                dr_file,
                bh_mass=bh_mass / 1e6,
                output_dir=os.path.join(OUTPUT_DIR, "plots", sim_fname),
            )


if __name__ == '__main__':
    SIMULATION_FILES_LIST = glob.glob(
        "/media3/rupesh/crp8/code/simulation_runs/crp8_variability/tde_simulation/ngc_9_5mpc_tide_lcs/*.csv")

    with Pool(processes=1) as pool:
        pool.map(run_pipeline, SIMULATION_FILES_LIST)
