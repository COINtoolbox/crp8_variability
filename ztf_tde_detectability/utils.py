
import os
import json
from typing import Any, Dict, List, Tuple, Optional


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ZTF_MAG_LIMIT: float = 23.5
SIGMA_THRESH: float = 5.0
LOG10_FACTOR: float = 1.0857  # 2.5 / ln(10)
MIN_FLUX: float = 1e-12


def mag_to_flux(mag: np.ndarray) -> np.ndarray:
    mag = np.asarray(mag, dtype=float)
    return 10 ** (26 - (mag + 48.6) / 2.5)


def flux_to_mag(flux: np.ndarray) -> np.ndarray:
    flux = np.asarray(flux, dtype=float)

    f_cgs = flux * 1e-3 * 1e-23
    return -2.5 * np.log10(f_cgs) - 48.6


def distance_suffix(distance: float) -> str:
    return f"_{distance:g}".replace(".", "p")


def shift_magnitude_to_distance(
    mag: np.ndarray,
    d_old: float,
    d_new: float,
) -> np.ndarray:
    mag = np.asarray(mag, dtype=float)
    delta_mu = 5.0 * np.log10(d_new / d_old)
    return mag + delta_mu


def compute_reference_mJy(filter_df: pd.DataFrame):
    flux: np.ndarray = mag_to_flux(filter_df["mag"].to_numpy())
    n: int = flux.size

    if n == 0:
        return np.nan, np.nan, np.nan, np.nan

    ref_flux: float = np.median(flux)

    mad: float = np.median(np.abs(flux - ref_flux))
    ref_flux_err: float = 1.4826 * mad / np.sqrt(n)

    ref_mag: float = flux_to_mag(ref_flux)
    ref_mag_err: float = (2.5 / np.log(10)) * (ref_flux_err / ref_flux)

    return ref_flux, ref_flux_err, ref_mag, ref_mag_err


def compute_snr(sigmapsf: np.ndarray) -> np.ndarray:
    sigmapsf = np.asarray(sigmapsf)

    valid = pd.notna(sigmapsf) & (sigmapsf > 0)

    snr = np.zeros_like(sigmapsf, dtype=float)
    snr[valid] = LOG10_FACTOR / sigmapsf[valid]

    return snr


def is_trigger_alert(
    snr: np.ndarray,
    idp_converted: np.ndarray,
    mjd_column: np.ndarray,
    start_mjd: float,
) -> np.ndarray:
    return (
        (snr > SIGMA_THRESH)
        & (idp_converted == 1)
        & (mjd_column >= start_mjd)
    )


def dc_to_difference_mag(
    dc_mag: float,
    dc_sigmag: float,
    magnr: float,
    sigmagnr: float,
    clip_limit: float = ZTF_MAG_LIMIT,
) -> Tuple[float, float, int]:
    if pd.isna(dc_mag) or pd.isna(magnr) or magnr < 0:
        return np.nan, np.nan, 0

    flux_dc: float = 10 ** (-0.4 * dc_mag)
    flux_ref: float = 10 ** (-0.4 * magnr)

    sigf_dc: float = (dc_sigmag / LOG10_FACTOR) * flux_dc
    sigf_ref: float = (sigmagnr / LOG10_FACTOR) * flux_ref

    flux_diff: float = flux_dc - flux_ref
    isdiffpos: int = int(flux_diff >= 0)
    flux_diff = abs(flux_diff)

    if flux_diff <= MIN_FLUX:
        return np.nan, np.nan, isdiffpos

    sigf_diff: float = np.hypot(sigf_dc, sigf_ref)

    magpsf: float = -2.5 * np.log10(flux_diff)
    sigmapsf: float = (sigf_diff / flux_diff) * LOG10_FACTOR

    if magpsf > clip_limit:
        return np.nan, np.nan, isdiffpos

    return magpsf, sigmapsf, isdiffpos


def load_snad_data(json_file: str) -> pd.DataFrame:
    with open(json_file, "r") as f:
        data: Dict[str, Any] = json.load(f)

    rows: List[Dict[str, Any]] = []

    for obj_id, obj in data.items():
        meta: Dict[str, Any] = obj.get("meta", {})
        lc: List[Dict[str, Any]] = obj.get("lc", [])

        coord: Dict[str, Any] = meta.get("coord", {})
        ra: float = coord.get("ra", np.nan)
        dec: float = coord.get("dec", np.nan)
        filt: str = meta.get("filter", "")

        for point in lc:
            rows.append({
                "oid": obj_id,
                "ra": ra,
                "dec": dec,
                "mjd": point.get("mjd", np.nan),
                "mag": point.get("mag", np.nan),
                "magerr": point.get("magerr", np.nan),
                "filtercode": filt,
            })

    return pd.DataFrame.from_records(rows)


def add_observation_noise(
    flux: np.ndarray,
    magerr: np.ndarray,
    simulated_flux: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flux_err = (magerr / LOG10_FACTOR) * flux
    noise = np.random.normal(0.0, flux_err)
    noisy_flux = simulated_flux + noise
    return noisy_flux, flux_err


def simulate_flux_with_tide(
    real_df: pd.DataFrame,
    tide_df: pd.DataFrame,
    start_mjd: float,
    flux_column_name: str,
) -> pd.DataFrame:

    if len(real_df) < 2:
        return pd.DataFrame()

    real_df = real_df.copy()
    tide_df = tide_df.sort_values("time_days")

    model_times = start_mjd + tide_df["time_days"].to_numpy()
    model_flux = tide_df[flux_column_name].to_numpy()

    obs_times = real_df["mjd"].to_numpy()

    t_min, t_max = model_times.min(), model_times.max()
    valid_mask = (obs_times >= t_min) & (obs_times <= t_max)

    if not np.any(valid_mask):
        print("No overlap between model and data.")

        flux = real_df["flux"].to_numpy()
        magerr = real_df["magerr"].to_numpy()

        real_df["simulated_flux"] = flux
        real_df["simulated_flux_err"] = (magerr / 1.0857) * flux
        real_df["simulated_mag_error"] = magerr
        real_df["simulated_mag"] = flux_to_mag(flux)
        real_df["original_mag"] = flux_to_mag(flux)
        real_df["overlap_found"] = False
        return real_df



    updated_flux = real_df["flux"].to_numpy().copy()

    # interpolate model contribution
    flux_diff = np.interp(obs_times[valid_mask], model_times, model_flux)
    updated_flux[valid_mask] += flux_diff

    noisy_flux, flux_err = add_observation_noise(
        real_df["flux"].to_numpy(),
        real_df["magerr"].to_numpy(),
        updated_flux,
    )

    real_df["simulated_flux"] = noisy_flux
    real_df["simulated_flux_err"] = flux_err

    real_df["simulated_mag_error"] = (flux_err / np.abs(noisy_flux)) * 1.0857
    real_df["simulated_mag"] = flux_to_mag(noisy_flux)
    real_df["original_mag"] = flux_to_mag(real_df["flux"].to_numpy())
    real_df["overlap_found"] = True

    return real_df


def plot_combined_mags_per_distance(
    object_sim_results: pd.DataFrame,
    each_dr_file: str,
    magnr_dict: Optional[Dict[str, float]] = None,
    bh_mass: float = 0,
    output_dir: str = "plots",
) -> None:

    os.makedirs(output_dir, exist_ok=True)

    mag_cols: list[str] = [
        c for c in object_sim_results.columns
        if c.startswith("simulated_mag") and "error" not in c
    ]
    distance_suffixes: list[str] = [
        c.replace("simulated_mag", "") for c in mag_cols
    ]

    n_distances: int = len(distance_suffixes)

    fig, axes = plt.subplots(
        n_distances,
        2,
        figsize=(16, 4.5 * n_distances),
        sharex=True,
    )

    if n_distances == 1:
        axes = np.array([axes])

    colors: Dict[str, str] = {
        "zg": "#141E3C",
        "zr": "#FF5B0B",
        "zi": "orange",
    }

    grouped: Dict[str, pd.DataFrame] = {
        fid: df.copy()
        for fid, df in object_sim_results.groupby("filter")
    }

    for i, suffix in enumerate(distance_suffixes):
        ax_dc, ax_diff = axes[i]

        for fid, df in grouped.items():
            color: str = colors.get(fid, "black")

            dc_col: str = f"simulated_mag{suffix}"
            dc_err_col: str = f"simulated_mag_error{suffix}"
            diff_col: str = f"magpsf{suffix}"
            diff_err_col: str = f"sigmapsf{suffix}"
            alert_col: str = f"is_alert{suffix}"
            dimmer_col: str = f"is_dimmer{suffix}"

            if dc_col not in df.columns or diff_col not in df.columns:
                continue

            bright: pd.DataFrame = df[~df[dimmer_col]]
            dim: pd.DataFrame = df[df[dimmer_col]]


            ax_dc.errorbar(
                bright["mjd"], bright[dc_col],
                yerr=bright.get(dc_err_col),
                fmt="o",
                color=color,
                alpha=0.6,
                markersize=4,
                elinewidth=0.7,
                capsize=0,
                markeredgecolor="none",
                label=fid,
            )

            ax_dc.errorbar(
                dim["mjd"], dim[dc_col],
                yerr=dim.get(dc_err_col),
                fmt="o",
                color=color,
                alpha=0.15,
                markersize=3,
                elinewidth=0.5,
                capsize=0,
                markeredgecolor="none",
            )


            ax_diff.errorbar(
                bright["mjd"], bright[diff_col],
                yerr=bright.get(diff_err_col),
                fmt="o",
                color=color,
                alpha=0.6,
                markersize=4,
                elinewidth=0.7,
                capsize=0,
                markeredgecolor="none",
                label=fid,
            )

            ax_diff.errorbar(
                dim["mjd"], dim[diff_col],
                yerr=dim.get(diff_err_col),
                fmt="o",
                color="grey",
                alpha=0.2,
                markersize=3,
                elinewidth=0.5,
                capsize=0,
                markeredgecolor="none",
            )

            if alert_col in df.columns:
                alerts: pd.DataFrame = df[df[alert_col]]

                if not alerts.empty:
                    ax_dc.scatter(
                        alerts["mjd"], alerts[dc_col],
                        facecolors="none",
                        edgecolors=color,
                        s=60,
                        linewidths=0.8,
                        alpha=0.6,
                    )

                    ax_diff.scatter(
                        alerts["mjd"], alerts[diff_col],
                        facecolors="none",
                        edgecolors=color,
                        s=60,
                        linewidths=0.8,
                        alpha=0.6,
                    )


            if magnr_dict and fid in magnr_dict:
                ax_dc.axhline(
                    magnr_dict[fid],
                    color=color,
                    linestyle="--",
                    alpha=0.25,
                    linewidth=1,
                )

        dist_label: str = suffix.replace("_", "").replace("p", ".") or "Original"

        ax_dc.set_ylabel(f"DR Mag\n{dist_label} Mpc")
        ax_dc.invert_yaxis()
        ax_dc.grid(True, linestyle="--", alpha=0.6)

        ax_diff.set_ylabel("Difference Mag")
        ax_diff.invert_yaxis()
        ax_diff.grid(True, linestyle="--", alpha=0.6)

        if i == 0:
            ax_dc.set_title("Simulated Light Curve (DR)")
            ax_diff.set_title("Difference Imaging")

        for ax in (ax_dc, ax_diff):
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(
                by_label.values(),
                by_label.keys(),
                fontsize=8,
                frameon=False,
            )

    axes[-1, 0].set_xlabel("MJD")
    axes[-1, 1].set_xlabel("MJD")

    file_tag: str = os.path.splitext(os.path.basename(each_dr_file))[0]

    plt.suptitle(
        f"ZTF IMBH TDE Simulation: {file_tag}  "
        f"$M_{{BH}} = {bh_mass} \\times 10^6 M_\\odot$",
        fontsize=13,
        y=0.995,
    )

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/{file_tag}_styled.png",
        dpi=300,
        bbox_inches="tight",
    )
