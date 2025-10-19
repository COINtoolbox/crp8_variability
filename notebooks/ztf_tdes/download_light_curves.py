"""
    Download and process light curves from Fink and IRSA for a list of TDEs.
"""

import os
import io
import pandas as pd
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm
from fink_utils.photometry.conversion import dc_mag


INPUT_CSV = "tdes_list.csv"
OUTPUT_DIR = "data"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
SEARCH_RADIUS_ARCSEC = 1.2

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def get_fink_data(object_id):
    """Retrieve light curve data from the Fink broker.

    Args:
        object_id (str): ZTF object ID.

    Returns:
        pd.DataFrame: Fink light curve data.
    """
    url = "https://api.fink-portal.org/api/v1/objects"
    payload = {
        "objectId": object_id,
        "output-format": "json"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return pd.read_json(io.BytesIO(response.content))
    except Exception as error:
        print(f"[Fink] Error for {object_id}: {error}")
        return pd.DataFrame()


def get_irsa_data(ra, dec):
    """Retrieve light curve data from IRSA for a given RA/Dec.

    Args:
        ra (float): Right Ascension in degrees.
        dec (float): Declination in degrees.

    Returns:
        pd.DataFrame: IRSA light curve data.
    """
    url = (
        "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves?"
        f"POS=CIRCLE+{ra}+{dec}+{SEARCH_RADIUS_ARCSEC / 3600.0}&FORMAT=csv"
    )

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), comment="#")
        df = df[df["catflags"] == 0]

        if df.empty:
            return pd.DataFrame()

        df["filter"] = df["filtercode"].map({"zg": "g", "zr": "r", "zi": "i"})
        df["source"] = "irsa"
        return df[["mjd", "mag", "magerr", "filter", "source"]]

    except Exception as error:
        print(f"[IRSA] Error for RA={ra}, Dec={dec}: {error}")
        return pd.DataFrame()


def process_fink_df(df):
    """Clean and standardize Fink data.

    Args:
        df (pd.DataFrame): Raw Fink data.

    Returns:
        pd.DataFrame: Cleaned and formatted light curve data.
    """
    if df.empty:
        return df

    df = df[
        df["i:magpsf"].notna() &
        df["i:magnr"].notna() &
        df["i:isdiffpos"].notna()
    ].copy()

    if df.empty:
        return df

    df["mjd"] = df["i:jd"] - 2400000.5
    df["filter"] = df["i:fid"].map({1: "g", 2: "r"})

    df[["mag", "magerr"]] = df.apply(
        lambda row: pd.Series(dc_mag(
            magpsf=row["i:magpsf"],
            sigmapsf=row["i:sigmapsf"],
            sigmagnr=row["i:sigmagnr"],
            magnr=row["i:magnr"],
            isdiffpos=row["i:isdiffpos"]
        )),
        axis=1
    )

    df["source"] = "fink"
    return df[["mjd", "mag", "magerr", "filter", "source"]]


def plot_lightcurve(df, object_id, is_plot=False):
    """Plot and save a light curve from combined Fink and IRSA data.

    Args:
        df (pd.DataFrame): Merged light curve data.
        object_id (str): Identifier for the object.
    """
    if is_plot:
        plt.figure(figsize=(10, 6))
        for filt in ["g", "r", "i"]:
            for source in ["fink", "irsa"]:
                data = df[(df["filter"] == filt) & (df["source"] == source)]
                if not data.empty:
                    plt.errorbar(
                        data["mjd"],
                        data["mag"],
                        yerr=data["magerr"],
                        fmt="o",
                        label=f"{filt} ({source})",
                        alpha=0.7
                    )

        plt.gca().invert_yaxis()
        plt.xlabel("MJD")
        plt.ylabel("Apparent Magnitude")
        plt.title(f"Light Curve: {object_id}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join(PLOT_DIR, f"{object_id}_lightcurve.png")
        plt.savefig(plot_path)
        plt.close()


def main():
    df_targets = pd.read_csv(INPUT_CSV)

    for _, row in tqdm(df_targets.iterrows(), total=len(df_targets)):
        object_id = row["objectId"]
        ra = row["ra"]
        dec = row["decl"]
        output_file = os.path.join(OUTPUT_DIR, f"{object_id}.csv")

        if os.path.isfile(output_file):
            continue

        df_fink_raw = get_fink_data(object_id)
        df_fink = process_fink_df(df_fink_raw)
        df_irsa = get_irsa_data(ra, dec)

        df_all = pd.concat([df_fink, df_irsa], ignore_index=True)
        df_all = df_all.sort_values("mjd")

        df_all.to_csv(output_file, index=False)
        plot_lightcurve(df_all, object_id)


if __name__ == "__main__":
    main()
