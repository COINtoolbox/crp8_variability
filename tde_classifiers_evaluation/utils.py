"""
    Utils
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from fink_filters.ztf.filter_early_tde_candidates.prefilter import mag2fluxcal
from fink_filters.ztf.filter_early_tde_candidates import lcs


BAND_FILTER_MAPPING = {
    "zg": 1,
    "zr": 2,
}

BAND_COLORS = {
    "zg": "#141E3C",
    "zr": "#FF5B0B",
    "zi": "orange",
}

MODEL_FILES = [
    "model_nuclear.ubj",
    "model_broad.ubj",
]


def plot_light_curves(df: pd.DataFrame, object_id: str) -> None:
    """Plot original and simulated light curves."""
    _, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    _plot_band_scatter(
        axes[0],
        df,
        y_col="original_mag",
        title=f"Original Light Curve\nObject ID: {object_id}",
    )

    _plot_band_scatter(
        axes[1],
        df,
        y_col="simulated_mag",
        title="Simulated Light Curve",
        marker="s",
        alpha=0.6,
    )

    axes[1].set_xlabel("MJD")
    plt.tight_layout()
    plt.show()


def _plot_band_scatter(
    ax,
    df: pd.DataFrame,
    y_col: str,
    title: str,
    marker: str = "o",
    alpha: float = 0.4,
) -> None:
    for fid in df["i:fid"].unique():
        band_data = df[df["i:fid"] == fid]
        ax.scatter(
            band_data["i:jd"],
            band_data[y_col],
            color=BAND_COLORS.get(fid, "black"),
            label=f"{y_col.replace('_', ' ').title()} {fid}",
            marker=marker,
            alpha=alpha,
            s=40,
        )

    ax.invert_yaxis()
    ax.set_ylabel("AB Magnitude")
    ax.set_title(title)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)


def preprocess_and_extract_features(
    df: pd.DataFrame,
    sampling_window: int = 100,
    history_window: int = 100,
):
    df = df.copy()

    df["FLUXCAL"], df["FLUXCALERR"] = mag2fluxcal(
        df["simulated_mag"], df["magerr"]
    )
    df["FLUXCALUPPER"] = 10 ** (11 - 0.4 * df["limitmag"])

    df["i:fid"] = df["i:fid"].map(BAND_FILTER_MAPPING)
    df = df.sort_values("i:jd").reset_index(drop=True)

    lcs.deredden_pdf(df, np.nanmean(df["ra"]), np.nanmean(df["dec"]))

    jd_max = df["i:jd"].max()
    jd_min = jd_max - sampling_window

    sampled_lc = df[(df["i:jd"] >= jd_min) & (df["i:jd"] <= jd_max)]

    if not _has_required_bands(sampled_lc):
        return None, sampled_lc

    if not _passes_history_checks(df, jd_min, history_window):
        return None, sampled_lc

    sampled_lc = lcs.cleanup_limits(sampled_lc)
    features = lcs.extract_features(sampled_lc, nsamples=1000)

    return features, sampled_lc


def _has_required_bands(df: pd.DataFrame) -> bool:
    return set(BAND_FILTER_MAPPING.values()).issubset(df["i:fid"].unique())


def _passes_history_checks(
    df: pd.DataFrame,
    jd_min: float,
    history_window: int,
) -> bool:
    history_mask = (df["i:jd"] <= jd_min) & (
            df["i:jd"] > jd_min - history_window)
    pre_history_mask = df["i:jd"] <= jd_min

    num_history_detections = np.sum(
        history_mask & np.isfinite(df["FLUXCAL"])
    )
    num_pre_history_negatives = np.sum(
        pre_history_mask & np.isfinite(df["FLUXCAL"]) & (df["FLUXCAL"] < 0)
    )

    return num_history_detections == 0 and num_pre_history_negatives <= 1


def update_features(
    lc_features: dict,
    object_df: pd.DataFrame,
    distnr: float,
) -> dict:
    jd_max = object_df["i:jd"].max()
    results = {}

    for i, feature_name in enumerate(lcs.feature.names):
        name = "temperature" if feature_name == "T" else feature_name

        values = [lc_features["params"][i]] + [
            sample[i] for sample in lc_features["samples"]
        ]

        results[name] = np.array(values)
        results[f"e_{name}"] = lc_features["errors"][i]

        if name != "reference_time":
            results[f"snr_{name}"] = abs(
                lc_features["params"][i] / lc_features["errors"][i]
            )

    results["r_chisq"] = lc_features["params"][-1]
    results["rel_reference_time"] = results["reference_time"] - jd_max
    results["norm_rel_reference_time"] = (
        results["rel_reference_time"] / results["rise_time"]
    )
    results["distnr"] = distnr

    return results


def check_features_quality(features: dict) -> bool:
    return not (
        features["r_chisq"] > 10
        or features["e_reference_time"] > 100
        or features["norm_rel_reference_time"][0] > 1
        or features["norm_rel_reference_time"][0] < -10
        or features["snr_rise_time"] < 1.5
        or features["snr_temperature"] < 1.5
    )


def load_classifiers(fink_filters_path):
    classifiers = []
    model_dir = (fink_filters_path / "fink_filters" / "ztf" /
                 "filter_early_tde_candidates" / "data")

    for model_file in MODEL_FILES:
        clf = xgb.XGBClassifier()
        clf.load_model(model_dir / model_file)
        classifiers.append(clf)

    return classifiers


def run_classifier(
    lc_features: dict,
    classifiers: list[xgb.XGBClassifier],
    object_id: str,
) -> dict:
    results = {
        "objectId": object_id,
        "best_score": [],
        "frac_scores": [],
        "valid": True,
    }

    feature_df = pd.DataFrame(lc_features)

    for each_classifier in classifiers:
        features = feature_df[each_classifier.feature_names_in_]
        scores = each_classifier.predict(features)

        best_score = scores[0]
        frac_score = scores.mean()

        results["best_score"].append(best_score)
        results["frac_scores"].append(frac_score)

        if not best_score and frac_score < 0.1:
            results["valid"] = False

    return results
