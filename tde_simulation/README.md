## Simulation of IMBH caused TDEs in ZTF lightcurves

The pipeline simulates ZTF observation photometry lightcurves by mapping theoretical simulations generated with TiDE ( [Kovács-Stermeczky & Vinkó (2023) ](https://ui.adsabs.harvard.edu/abs/2023PASP..135c4102K/abstract)) into real observation.



### Requirements:
- Install TiDE https://github.com/stermzsofi/TiDE
- Install dependencies in requirements.txt

 To run the script
```bash
python run_simulation.py --config config.json
```
### Simulation Configuration (`config.json`)

The example configuration with default parameters is available in `config.json`
- **ztf_files_path**:  
 Path to ZTF data release photometry downloaded from IRSA. Check the example input format in the data folder.
  *Example:* `/path/to/data/*.csv`

- **tide_path**:  
  Path to TiDE installation directory.  

- **output_dir**:  
  Path to output directory to save simiulation data, config and plots.  

- **save_plots**:  
  When set to True simulated output path will be saved in output folder

- **filters**:  
  Filters data to simulate
  *Example:* `["zg", "zr"]`

#### TiDE related configs
- **bh_mass**:  
  Black hole masses (in M☉)  range to sample for simulations.  
  - `min`: minimum black hole mass (`1e2`)  
  - `max`: maximum black hole mass (`1e6`)  

- **star_mstar**:  
  stellar masses (in M☉) for disrupted stars.  
  - `min`: minimum stellar mass (`0.1`)  
  - `max`: maximum stellar mass (`30`)  

- **star_rstar**:  
  Stellar radius distribution settings (categorical, continuous or mixed)
  - `categorical`: stellar types to include (`["ms", "wd"]` = main sequence and white dwarfs)  
  - `continuous`: range of stellar radii in solar radii (`min: 0.1, max: 10.0`)
  - `mixed`: Can sample from categorical and continuous

- **use_p_nu_scaling**:  
   List to booleans indicating whether to apply TiDE p.nu scaling in simulations.  
  *Example:* `[true]`

- **distance_mpc**:  
  Either this or the `redshift` value should be provided. Distance to simulated TDEs in megaparsecs. Used to compute apparent flux.  
  *Example:* `100`

- **redshift**:  
  Either this or the `distnce_mpc` should be provided. Redshift value used in computing comoving distance,
which is used as distanceto compute apparent flux.  
  *Example:* `0.5`
