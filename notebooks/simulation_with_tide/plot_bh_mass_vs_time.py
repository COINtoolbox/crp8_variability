"""
    Plot blackhole mass vs flare timescale from TiDE
"""
import tidepy
import numpy as np
import matplotlib.pyplot as plt

bh_masses_solar = np.logspace(2, 6, 15)
bh_masses_M6 = bh_masses_solar / 1e6

t_peaks = []

TPEAK_RELATIVE_TO_TMIN_N3 = 5.77316

for bh_mass_M6 in bh_masses_M6:
    p = tidepy.Parameters()
    p.bh_M6 = bh_mass_M6
    p.param_init()

    t_min = tidepy.tide.lib.get_parameters_tmin(p.param)
    t_peak = t_min * TPEAK_RELATIVE_TO_TMIN_N3
    t_peaks.append(t_peak)

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(bh_masses_solar, t_peaks, marker='s', linestyle='-', color='darkred')
ax.set_xlabel(r'Black Hole Mass [$M_\odot$]')
ax.set_ylabel(r'Peak Time ($t_{\rm peak}$) [days]')
ax.set_title('Time to Peak vs. Black Hole Mass')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

lower_bound = 13.24 - 3.68
upper_bound = 13.24 + 3.68
ax.axhspan(lower_bound, upper_bound, color='gray', alpha=0.3, label='13.24 ± 3.68 days')

bh_mass_target = 1e6
idx_closest = np.abs(bh_masses_solar - bh_mass_target).argmin()
t_peak_at_1e6 = t_peaks[idx_closest]
ax.plot(bh_mass_target, t_peak_at_1e6, marker='x', color='black', markersize=10, label='SMBH')
ax.annotate('SMBH', xy=(bh_mass_target, t_peak_at_1e6),
            xytext=(bh_mass_target * 0.6, t_peak_at_1e6 + 10),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=10)

ax.set_xscale('log')
ax.legend()
plt.tight_layout()
plt.savefig("bh_mass_vs_tpeak_plot.png")
plt.show()
