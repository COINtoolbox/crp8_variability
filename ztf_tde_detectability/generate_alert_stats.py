import os
import glob
import json
import numpy as np

import pandas as pd
from tqdm import tqdm


def extract_distances(columns):
    return [
        c.replace("simulated_mag", "")
        for c in columns
        if c.startswith("simulated_mag") and "error" not in c
    ]


def process_file(file_path):
    df = pd.read_csv(file_path)

    object_id = df["object_id"].iloc[0]
    bh_mass = df["bh_mass_msun"].iloc[0]
    start_mjd = df["start_mjd"].iloc[0]

    distances = [
        c.replace("simulated_mag", "")
        for c in df.columns
        if c.startswith("simulated_mag") and "error" not in c
    ]

    result = {}

    for suffix in distances:
        dist_key = suffix.replace("_", "").replace("p", ".")

        mag_col = f"simulated_mag{suffix}"
        alert_col = f"is_alert{suffix}"
        snr_col = f"snr{suffix}"

        if mag_col not in df.columns:
            continue

        result[dist_key] = {}

        for filt in df["filter"].unique():
            sub = df[df["filter"] == filt].copy()

            sub = sub.dropna(subset=[mag_col])
            if sub.empty:
                continue

            peak_idx = sub[mag_col].idxmin()

            result[dist_key][filt] = {
                "n_alerts": int(sub[alert_col].sum()) if alert_col in sub else 0,
                "n_points": int(len(sub)),
                "peak_mag": float(sub.loc[peak_idx, mag_col]),
                "peak_mjd": float(sub.loc[peak_idx, "mjd"]),
                "mean_snr": float(sub[snr_col].mean()) if snr_col in sub else None
            }

    return bh_mass, object_id, start_mjd, result



def json_to_dataframe(summary_json):
    rows = []

    for bh_mass, objects in summary_json.items():
        for obj_id, obj_data in objects.items():
            start_mjd = obj_data["start_mjd"]
            for dist, filters in obj_data["distances"].items():
                for filt, stats in filters.items():
                    row = {
                        "bh_mass": float(bh_mass),
                        "object_id": obj_id,
                        "distance": float(dist),
                        "filter": filt,
                        "start_mjd": start_mjd,
                        **stats
                    }
                    rows.append(row)

    return pd.DataFrame(rows)


def convert_keys_and_values(obj):

    if isinstance(obj, dict):
        return {str(k): convert_keys_and_values(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_keys_and_values(v) for v in obj]
    else:
        return obj


def build_summary_json(input_dir, output_json="summary.json"):
    all_files = glob.glob(os.path.join(input_dir, "*/*.csv"))

    final_dict = {}

    for file in tqdm(all_files):
        result = process_file(file)
        if result is None:
            continue

        bh_mass, object_id, start_mjd, dist_data = result

        bh_key = str(bh_mass)

        if bh_key not in final_dict:
            final_dict[bh_key] = {}

        final_dict[bh_key][object_id] = {
            "start_mjd": start_mjd,
            "distances": {str(k): v for k, v in dist_data.items()}
        }
    final_df = json_to_dataframe(final_dict)
    output_csv = output_json.replace(".json", ".csv")
    final_df.to_csv(output_csv, index=False)
    final_dict = convert_keys_and_values(final_dict)

    with open(output_json, "w") as f:
        json.dump(final_dict, f, indent=2)

    print(f"Saved to {output_json}")


input_dir = "/media3/rupesh/crp8/data/legus_photometry/snad_simulation_fixed_final_100mpc"
output_json =  "/media3/rupesh/crp8/data/legus_photometry/snad_simulation_fixed_final_100mpc/output_final_100mpc.json"
print(input_dir)
print(output_json)
build_summary_json(input_dir=input_dir, output_json=output_json)
