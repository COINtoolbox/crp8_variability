"""
    Simulate TDEs with TiDE package
"""

import json
import tidepy
import numpy as np
import pandas as pd
import random
import os
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from typing import Optional, Dict, Any, Union, List


def redshift_to_cm(z: float) -> float:
    """Convert redshift to comoving distance in centimeters.

    Args:
        z: Redshift value.

    Returns:
        Comoving distance in centimeters.
    """
    return cosmo.comoving_distance(z).to(u.cm).value


def mpc_to_cm(mpc: float) -> float:
    """Convert megaparsecs to centimeters.

    Args:
        mpc: Distance in megaparsecs.

    Returns:
        Distance in centimeters.
    """
    return mpc * 3.0857e24


def luminosity_to_flux_mjy(L: np.ndarray,
                           d_cm: float,
                           p_nu: Optional[float] = None) -> np.ndarray:
    """Convert luminosity to flux in millijanskys.

    Args:
        L: Luminosity (array).
        d_cm: Distance in centimeters.
        p_nu: Optional scaling factor.

    Returns:
        Flux in millijanskys.
    """
    flux = L / (4 * np.pi * d_cm**2)
    if p_nu is not None:
        flux /= p_nu
    return flux / 1e-26


def flux_to_mag(flux_mjy: np.ndarray) -> np.ndarray:
    """Convert flux in millijanskys to magnitude.

    Args:
        flux_mjy: Flux in millijanskys.

    Returns:
        Magnitude (array).
    """
    return -2.5 * np.log10(flux_mjy * 1e-26) - 48.6


def load_config(path: str) -> Dict[str, Any]:
    """Load a JSON configuration file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(path, "r") as f:
        return json.load(f)


def sample_star_rstar(cfg: Dict[str, Any]) -> Union[float, int]:
    """Sample star radius based on configuration.

    Args:
        cfg: Configuration dictionary with mode and parameters.

    Raises:
        ValueError: If mode is invalid.

    Returns:
        Sampled star radius value.
    """
    mode = cfg["mode"]
    if mode == "categorical":
        return random.choice(cfg["categorical"])
    elif mode == "continuous":
        return round(np.random.uniform(cfg["continuous"]["min"],
                                      cfg["continuous"]["max"]), 2)
    elif mode == "mixed":
        if random.random() < 0.5:
            return random.choice(cfg["categorical"])
        else:
            return round(np.random.uniform(cfg["continuous"]["min"],
                                          cfg["continuous"]["max"]), 2)
    else:
        raise ValueError(f"Invalid star_rstar mode: {mode}")


def generate_lightcurve(sample_id: int,
                        bh_mass: float,
                        star_mass: float,
                        star_rstar: Union[float, int],
                        use_p_nu_scaling: bool,
                        output_dir: str,
                        distance_mpc: Optional[float] = None,
                        redshift: Optional[float] = None
                        ) -> Dict[str, Union[str, float, bool, None]]:
    """Generate a light curve for a tidal disruption event (TDE) sample.

    Args:
        sample_id: Sample identifier.
        bh_mass: Black hole mass.
        star_mass: Star mass.
        star_rstar: Star radius.
        use_p_nu_scaling: Whether to use p.nu scaling.
        output_dir: Directory to save output CSV.
        distance_mpc: Distance in megaparsecs (optional).
        redshift: Redshift (optional).

    Raises:
        ValueError: If both or neither distance_mpc and redshift are specified.

    Returns:
        Dictionary of sample parameters.
    """
    if (distance_mpc is None and redshift is None) or \
       (distance_mpc is not None and redshift is not None):
        raise ValueError("Specify either 'distance_mpc' or 'redshift', not both.")

    p = tidepy.Parameters()
    p.bh_M6 = bh_mass / 1e6
    p.star_mstar = star_mass
    p.star_rstar = star_rstar
    p.tend = 1000
    p.param_init()

    lc = tidepy.Light_curve_of_tde(p)
    results = lc.light_curve()
    time_days: np.ndarray = results[0]
    luminosity: np.ndarray = results[1]
    if use_p_nu_scaling:
        luminosity *= p.nu

    if redshift is not None:
        distance_cm = redshift_to_cm(redshift)
    else:
        # distance_mpc is guaranteed not None here
        distance_cm = mpc_to_cm(distance_mpc)  # type: ignore

    flux_mjy = luminosity_to_flux_mjy(
        luminosity, distance_cm, p.nu if use_p_nu_scaling else None)
    mag = flux_to_mag(flux_mjy)

    lc_df = pd.DataFrame({"time": time_days, "mag": mag})
    lc_filename = os.path.join(output_dir, f"sample_{sample_id:05d}.csv")
    lc_df.to_csv(lc_filename, index=False)

    return {
        "sample_id": f"{sample_id:05d}",
        "bh_mass": bh_mass,
        "star_mass": star_mass,
        "star_rstar": star_rstar,
        "use_p_nu_scaling": use_p_nu_scaling,
        "distance_mpc": distance_mpc,
        "redshift": redshift,
    }


def main() -> None:
    """Main function to run the TDE light curve simulation."""
    config = load_config(
        "config.json"
    )

    tide_path: Optional[str] = config.get("tide_path")
    if not tide_path:
        raise ValueError("Missing 'tide_path' in config.json")

    os.environ["TIDE_PATH"] = tide_path
    print(f"[INFO] TIDE_PATH set to: {tide_path}")

    output_dir: str = config.get("output_dir", tide_path)
    os.makedirs(output_dir, exist_ok=True)

    distance_mpc: Optional[float] = config.get("distance_mpc")
    redshift: Optional[float] = config.get("redshift")
    if (distance_mpc is None and redshift is None) or \
       (distance_mpc is not None and redshift is not None):
        raise ValueError(
            "Specify exactly one of 'redshift' or 'distance_mpc' in config.json")

    config_records: List[Dict[str, Union[str, float, bool, None]]] = []
    for i in range(1, config["num_samples"] + 1):
        bh_mass: float = np.random.uniform(config["bh_mass"]["min"],
                                          config["bh_mass"]["max"])
        star_mass: float = np.random.uniform(config["star_mass"]["min"],
                                            config["star_mass"]["max"])
        star_rstar = sample_star_rstar(config["star_rstar"])
        use_pnu: bool = random.choice(config["use_p_nu_scaling"])

        record = generate_lightcurve(
            sample_id=i,
            bh_mass=bh_mass,
            star_mass=star_mass,
            star_rstar=star_rstar,
            use_p_nu_scaling=use_pnu,
            output_dir=output_dir,
            distance_mpc=distance_mpc,
            redshift=redshift,
        )
        config_records.append(record)

        if i % 25 == 0 or i == config["num_samples"]:
            print(f"[INFO] Completed {i}/{config['num_samples']} samples")

    config_df = pd.DataFrame(config_records)
    config_df.to_csv(os.path.join(output_dir, "config.csv"), index=False)
    print(f"[INFO] Saved config to {os.path.join(output_dir, 'config.csv')}")


if __name__ == "__main__":
    main()
