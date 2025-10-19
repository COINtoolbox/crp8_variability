import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


INPUT_FOLDER = "/path/to/ztf_tdes/light_curve"
OUTPUT_FOLDER = "data/plots_with_peak_time"
SUMMARY_CSV_PATH = "data/flare_timescales_summary.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def find_peak_with_neighbors(df, time_window=2, mag_tolerance=0.2):
    """Finds the flare peak with neighboring points for validation.

    Args:
        df (pd.DataFrame): Light curve data containing 'mjd' and 'mag' columns.
        time_window (float): Time window in days to look for nearby points.
        mag_tolerance (float): Magnitude difference allowed for neighbors.

    Returns:
        tuple: (mjd_peak, mag_peak)
    """
    df_sorted = df.sort_values("mag").reset_index(drop=True)

    for _, row in df_sorted.iterrows():
        peak_mjd = row["mjd"]
        peak_mag = row["mag"]

        neighbors = df[
            (df["mjd"] >= peak_mjd - time_window) &
            (df["mjd"] <= peak_mjd + time_window)
        ]
        close_neighbors = neighbors[
            (np.abs(neighbors["mag"] - peak_mag) <= mag_tolerance) &
            (neighbors["mjd"] != peak_mjd)
        ]

        if not close_neighbors.empty:
            return peak_mjd, peak_mag

    return df_sorted.loc[0, "mjd"], df_sorted.loc[0, "mag"]  # fallback to global minimum


def analyze_lightcurve(file_path):
    """Processes a single light curve file to find the flare peak and plot it.

    Args:
        file_path (str): Path to the CSV light curve file.

    Returns:
        dict or None: Summary info including peak MJD and plot path, or None on failure.
    """
    object_id = os.path.basename(file_path).split("_")[0].replace(".csv", "")

    try:
        df = pd.read_csv(file_path)
        df = df[df["mag"].notna() & df["mjd"].notna()]
        df = df[df["filter"].isin(["g", "r"])]

        mjd_peak, mag_peak = find_peak_with_neighbors(df)

        # Plot full light curve with peak marked
        plt.figure(figsize=(10, 6))
        for filt in ["g", "r"]:
            for src, marker, alpha_val in [("fink", "o", 0.7), ("irsa", "s", 0.5)]:
                df_sub = df[(df["filter"] == filt) & (df["source"] == src)]
                if not df_sub.empty:
                    plt.errorbar(
                        df_sub["mjd"],
                        df_sub["mag"],
                        yerr=df_sub["magerr"],
                        fmt=marker,
                        label=f"{filt} ({src})",
                        alpha=alpha_val
                    )

        plt.axvline(mjd_peak, color="red", linestyle="--", label="Peak")
        plt.gca().invert_yaxis()
        plt.xlabel("MJD")
        plt.ylabel("Apparent Magnitude")
        plt.title(f"{object_id} Light Curve (g & r bands)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join(OUTPUT_FOLDER, f"{object_id}.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"{object_id}: peak MJD = {mjd_peak:.2f}")

        return {
            "object_id": object_id,
            "mjd_peak": mjd_peak,
            "mag_peak": mag_peak,
            "plot_path": plot_path,
        }

    except Exception as error:
        print(f"Failed to process {object_id}: {error}")
        return None


def main():
    """Main execution function to process all files and generate a summary."""
    results = []
    csv_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".csv")]
    print(f"Found {len(csv_files)} files in {INPUT_FOLDER}")

    for file_name in csv_files:
        file_path = os.path.join(INPUT_FOLDER, file_name)
        result = analyze_lightcurve(file_path)
        if result:
            results.append(result)

    if results:
        df_summary = pd.DataFrame(results)
        df_summary.to_csv(SUMMARY_CSV_PATH, index=False)
        print(f"Saved summary to {SUMMARY_CSV_PATH}")
    else:
        print("No valid results to save.")


if __name__ == "__main__":
    main()
