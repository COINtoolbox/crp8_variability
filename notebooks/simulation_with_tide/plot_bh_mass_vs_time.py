"""
    Plot blackhole mass vs flare timescale from TiDE
"""
import tidepy
import numpy as np
import matplotlib.pyplot as plt

bh_masses_solar = np.logspace(2, 6, 50)
bh_masses_M6 = bh_masses_solar / 1e6

flare_durations = []
t_peaks = []

TPEAK_RELATIVE_TO_TMIN_N3 = 5.77316

for bh_mass_M6 in bh_masses_M6:
    p = tidepy.Parameters()
    p.bh_M6 = bh_mass_M6
    p.param_init()

    t_min = tidepy.tide.lib.get_parameters_tmin(p.param)
    t_end = tidepy.tide.lib.get_parameters_tend(p.param)

    flare_duration = t_end - t_min
    t_peak = t_min * TPEAK_RELATIVE_TO_TMIN_N3

    flare_durations.append(flare_duration)
    t_peaks.append(t_peak)

fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

axs[0].plot(bh_masses_solar, flare_durations, marker='o', linestyle='-', color='navy')
axs[0].set_ylabel(r'Flare Timescale ($t_{\rm end} - t_{\rm min}$) [days]')
axs[0].set_title('Flare Duration vs. Black Hole Mass')
axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

axs[1].plot(bh_masses_solar, t_peaks, marker='s', linestyle='-', color='darkred')
axs[1].set_xlabel(r'Black Hole Mass [$M_\odot$]')
axs[1].set_ylabel(r'Peak Time ($t_{\rm peak}$) [days]')
axs[1].set_title('Time to Peak(tpeak_relative_to_tmin) vs. Black Hole Mass')
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig("bh_mass_1e2_to_1e6_vs_flare_and_peak_timescale.png")
plt.show()
