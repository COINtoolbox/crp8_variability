# Modified from:

# Copyright 2019-2022 AstroLab Software
# Authors: Marina Masson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd


def flux_to_mag(flux):
    """
    Convert flux from milliJansky to AB Magnitude

    1 Jy = 1e-23 erg/cm2/s/Hz
    Fnu = 3631 Jy = 3.631*1e-20 erg/cm2/s/Hz
    ABmag = 0-2.5*log10( Fnu )-48.6 = 0

    Parameters
    ----------
    flux: float
        Flux in milli-Jansky

    Returns
    -------
    mag: float
        Corresponding AB Magnitude
    """
    mag = -2.5 * np.log10(flux * 1.0e-26) - 48.6
    return mag


def mag_to_flux(mag):
    """
    Convert AB Magnitude to flux in milliJansky

    1 Jy = 1e-23 erg/cm2/s/Hz
    Fnu = 3631 Jy = 3.631*1e-20 erg/cm2/s/Hz
    ABmag = 0-2.5*log10( Fnu )-48.6 = 0

    Parameters
    ----------
    mag: float
        AB Magnitude

    Returns
    -------
    flux: float
        Corresponding flux in milliJansky
    """
    flux = pow(10, (26 - (mag + 48.6) / 2.5))
    return flux
