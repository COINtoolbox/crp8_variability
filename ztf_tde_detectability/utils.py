import os
import json
from typing import Any, Dict, List, Tuple, Union


from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
from astropy import units as u

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ZTF_MAG_LIMIT: float = 23.7
SIGMA_THRESH: float = 5.0
LOG10_FACTOR: float = 1.0857  # 2.5 / ln(10)
MIN_FLUX: float = 1e-12
PLOT_SIGMA_MAX: float = 0.75  # hide error bars above this σ_mag when plotting

FILTER_COLORS: Dict[str, str] = {
    "zg": "#141E3C",
    "zr": "#FF5B0B",
    "zi": "orange",
}


cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

def distance_to_z(d_mpc: float) -> float:
    return float(z_at_value(cosmo.luminosity_distance, d_mpc * u.Mpc))

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
    mag: Union[np.ndarray, float],
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
    sigma: float = 1.4826 * mad
    ref_flux_err: float = 1.2533 * sigma / np.sqrt(n)

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

    # Ignore the negative flux

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


def redraw_at_distance(
    injected_flux: np.ndarray,
    flux_err: np.ndarray,
    d_old: float,
    d_new: float,
) -> Tuple[np.ndarray, np.ndarray]:

    injected_flux = np.asarray(injected_flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    flux_dimmed = injected_flux * (d_old / d_new) ** 2
    sigma = np.where(np.isfinite(flux_err) & (flux_err > 0), flux_err, 0.0)
    noisy_flux = flux_dimmed + np.random.normal(0.0, sigma)

    mag = np.full(noisy_flux.shape, np.nan)
    magerr = np.full(noisy_flux.shape, np.nan)

    valid = noisy_flux > 0
    mag[valid] = flux_to_mag(noisy_flux[valid])
    magerr[valid] = (flux_err[valid] / noisy_flux[valid]) * LOG10_FACTOR

    return mag, magerr


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

    host_flux = real_df["flux"].to_numpy()
    magerr = real_df["magerr"].to_numpy()
    flux_err = (magerr / LOG10_FACTOR) * host_flux

    model_times = start_mjd + tide_df["time_days"].to_numpy()
    model_flux = tide_df[flux_column_name].to_numpy()
    obs_times = real_df["mjd"].to_numpy()

    t_min, t_max = model_times.min(), model_times.max()
    valid_mask = (obs_times >= t_min) & (obs_times <= t_max)

    injected_flux = host_flux.copy()
    if np.any(valid_mask):
        injected_flux[valid_mask] += np.interp(
            obs_times[valid_mask], model_times, model_flux
        )
        overlap = True
    else:
        print("No overlap between model and data.")
        overlap = False

    real_df["injected_flux"] = injected_flux
    real_df["flux_err"] = flux_err
    real_df["original_mag"] = flux_to_mag(host_flux)
    real_df["overlap_found"] = overlap

    return real_df


def _masked_err(err, sigma_max: float = PLOT_SIGMA_MAX):

    if err is None:
        return None
    err = np.asarray(err, dtype=float)
    out = err.copy()
    out[~(np.isfinite(err) & (err < sigma_max))] = np.nan
    return out


def _plot_filter_pair(
    ax_dc: plt.Axes,
    ax_diff: plt.Axes,
    df: pd.DataFrame,
    fid: str,
    suffix: str,
    color: str,
) -> None:
    dc_col       = f"simulated_mag{suffix}"
    dc_err_col   = f"simulated_mag_error{suffix}"
    diff_col     = f"magpsf{suffix}"
    diff_err_col = f"sigmapsf{suffix}"
    alert_col    = f"is_alert{suffix}"
    dimmer_col   = f"is_dimmer{suffix}"
    time_col     = f"simulated_mjd{suffix}"

    if dc_col not in df.columns or diff_col not in df.columns:
        return

    bright = df[~df[dimmer_col]]
    dim    = df[df[dimmer_col]]

    eb = dict(fmt="o", elinewidth=0.7, capsize=0, markeredgecolor="none")

    ax_dc.errorbar(
        bright[time_col], bright[dc_col], yerr=_masked_err(bright.get(dc_err_col)),
        color=color, alpha=0.6, markersize=4, label=fid, **eb,
    )
    ax_dc.errorbar(
        dim[time_col], dim[dc_col], yerr=_masked_err(dim.get(dc_err_col)),
        color=color, alpha=0.15, markersize=3, **eb,
    )

    ax_diff.errorbar(
        bright[time_col], bright[diff_col], yerr=_masked_err(bright.get(diff_err_col)),
        color=color, alpha=0.6, markersize=4, label=fid, **eb,
    )
    ax_diff.errorbar(
        dim[time_col], dim[diff_col], yerr=_masked_err(dim.get(diff_err_col)),
        color=color, alpha=0.2, markersize=3, **eb,
    )

    if alert_col in df.columns:
        alerts = df[df[alert_col]]
        if not alerts.empty:
            sc = dict(facecolors="none", edgecolors=color, s=60, linewidths=0.8, alpha=0.8)
            ax_dc.scatter(alerts[time_col], alerts[dc_col], **sc)
            ax_diff.scatter(alerts[time_col], alerts[diff_col], **sc)

    ref_col = f"reference_mag{suffix}"
    if ref_col in df.columns:
        ax_dc.axhline(df[ref_col].iloc[0], color=color, linestyle="--", alpha=0.25, linewidth=1)


def plot_combined_mags_per_distance(
    object_sim_results: pd.DataFrame,
    each_dr_file: str,
    bh_mass: float = 0,
    output_dir: str = "plots",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    mag_cols = [
        c for c in object_sim_results.columns
        if c.startswith("simulated_mag") and "error" not in c
    ]
    distance_suffixes = [c.replace("simulated_mag", "") for c in mag_cols]
    n_distances = len(distance_suffixes)

    fig, axes = plt.subplots(
        n_distances, 2,
        figsize=(16, 4.5 * n_distances),
        sharex=True,
    )
    if n_distances == 1:
        axes = np.array([axes])

    grouped = {fid: df.copy() for fid, df in object_sim_results.groupby("filter")}

    start_mjd = (
        object_sim_results["start_mjd"].iloc[0]
        if "start_mjd" in object_sim_results.columns else None
    )

    for i, suffix in enumerate(distance_suffixes):
        ax_dc, ax_diff = axes[i]

        for fid, df in grouped.items():
            _plot_filter_pair(ax_dc, ax_diff, df, fid, suffix, FILTER_COLORS.get(fid, "black"))

        if start_mjd is not None:
            for ax in (ax_dc, ax_diff):
                ax.axvline(start_mjd, color="grey", linestyle=":", linewidth=1, alpha=0.6, label="TDE start")

        dist_label = suffix.lstrip("_").replace("p", ".") or "Original"
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
            ax.legend(by_label.values(), by_label.keys(), fontsize=8, frameon=False)

    axes[-1, 0].set_xlabel("MJD")
    axes[-1, 1].set_xlabel("MJD")

    file_tag = os.path.splitext(os.path.basename(each_dr_file))[0]
    plt.suptitle(
        f"ZTF IMBH TDE Simulation: {file_tag}  "
        f"$M_{{BH}} = {bh_mass} \\times 10^6 M_\\odot$",
        fontsize=13,
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{file_tag}_styled.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
