"""
    Main inference script
"""


import json
import os.path
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import sys

from tde_classifiers_evaluation.utils import (
    preprocess_and_extract_features,
    update_features,
    check_features_quality,
    run_classifier,
    load_classifiers, plot_light_curves,
)

REQUIRED_FILTERS = {"zg", "zr"}

FINK_COLUMNS_MAPPING = {
    "filtercode": "i:fid",
    "mjd": "i:jd",
}

def load_classifier_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data(csv_path: Path, config_dir: Path):
    df = pd.read_csv(csv_path)

    object_config_path = config_dir / f"{csv_path.stem}.json"
    with open(object_config_path, "r", encoding="utf-8") as f:
        object_config = json.load(f)

    return df, object_config


def has_required_filters(df: pd.DataFrame) -> bool:
    return REQUIRED_FILTERS.issubset(df["filtercode"].unique())


def prepare_light_curve(df: pd.DataFrame, object_config: dict,
                        flare_rise_days) -> pd.DataFrame:
    df = df.rename(columns=FINK_COLUMNS_MAPPING)
    max_mjd = object_config["start_mjd"] + flare_rise_days
    return df[df["i:jd"] < max_mjd]


def process_object(
    csv_path: Path,
    classifiers: list,
    config_dir: Path,
    flare_rise_days,
    distnr: float,
    display_sampled_lc: bool = False
):
    object_id = csv_path.stem
    try:
        df, object_config = load_data(csv_path, config_dir)
    except:
        return "missing_config"
    if not has_required_filters(df):
        return "not_required_filters"

    df = prepare_light_curve(df, object_config, flare_rise_days)

    features_data, sampled_lc = preprocess_and_extract_features(df)
    if features_data is None:
        return "feature_extraction_failed"
    if display_sampled_lc:
        plot_light_curves(sampled_lc, object_id)

    features_data = update_features(features_data, df, distnr)

    if not check_features_quality(features_data):
        return "feature_quality_failed"
    return run_classifier(features_data, classifiers, object_id)


def main():
    pipeline_config = load_classifier_config(
        Path("classifier_config.json")
    )

    data_dir = Path(pipeline_config["data_dir"])
    config_dir = Path(pipeline_config["config_dir"])
    fink_filters_path = Path(pipeline_config["fink_filters_path"])
    flare_rise_days = pipeline_config["flare_rise_days"]
    object_distnr = pipeline_config["distnr"]
    classifiers = load_classifiers(fink_filters_path)
    results = []

    simulation_files = sorted(data_dir.glob("*.csv"))

    for csv_path in tqdm(simulation_files):
        result = process_object(
            csv_path,
            classifiers,
            config_dir,
            flare_rise_days,
            object_distnr
        )
        if isinstance(result, str):
            object_id = os.path.basename(csv_path).replace(".csv", "")
            results.append(
                {"objectId":object_id,
                 "best_score": [None],
                 "frac_scores": [None],
                 "valid": result})
        else:
            results.append(result)
        print(result)
    if results:
        pd.DataFrame(results).to_csv(
            "classification_results.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
