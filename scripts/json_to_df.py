import pandas as pd
import glob
import json
import argparse


def merge_json_to_df(list_files, save = True):
    all_lcs = []

    # Filter mapping
    filter_map = {"zg": ("g", 1), "zr": ("r", 2), "zi": ("i", 3)}
    for file in list_files:
        with open(os.path.join(folder, file), "r") as f:
            data = json.load(f)
        ztf_id = file.split("/")[-1].split(".")[0]
        
        # Loop over different matches (keys)
        for obj_id in data.keys():
            lc = data[obj_id]["lc"]
            filt = data[obj_id]["meta"]["filter"]  # e.g., "zg"
            
            if filt not in filter_map:
                continue 
            
            band, fid = filter_map[filt]
            
            # Extract points
            for point in lc:
                all_lcs.append({
                    "ztf_id": ztf_id,
                    "obj_id": obj_id,
                    "mjd": point["mjd"],
                    "mag": point["mag"],
                    "magerr": point["magerr"],
                    "band": band,
                    "fid": fid
                })

    # Convert to dataframe
    df = pd.DataFrame(all_lcs)
    if save:
        df.to_csv("merged_lightcurves.csv", index=False)
    else:
        return df
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--folder", type=str, required=True, help="Folder containing JSON")
    args = argparser.parse_args()
    folder = args.folder
    #folder = "/media3/CRP8/TDE/data/DR_photometry_full/extragalactic"

    list_files = glob.glob(f"{folder}/*.json")

    merge_json_to_df(list_files)