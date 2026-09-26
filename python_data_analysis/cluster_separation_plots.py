import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- TOGGLE FOR DISCRETE MODEL ANALYSIS ---
INCLUDE_DISCRETE_MODEL = False  # Set to False to exclude the discrete model
# ------------------------------------------

# 1. Load data from CSV
df = pd.read_csv('cluster_separation_data_v2.csv')

D_chem = df['D_chem'].values
lambda_max_exp = df['lambda_max'].values
lambda_max_err = df['lambda_max_err'].values

sqrt_D_chem = np.sqrt(D_chem)

# 2. Base simulation parameters
config = {
    "dx": 1.0,
    "width": 32,
    "height": 32,
    "decay_rate_gamma": 0.1,
    "chem_secretion_rate_alpha": 1.0,
    "total_cells": 10240000,
    "chi": 2.0,
    "Dr": 0.5  # D_cell
}

dx = config["dx"]
L = config["width"]  # Assuming square domain L = width = height
rho_0 = config["total_cells"] / (config["width"] * config["height"])
rho_c = (config["Dr"] * config["decay_rate_gamma"]) / (config["chi"] * config["chem_secretion_rate_alpha"])

# 3. Compute all available 2D Discrete Modes for a 32x32 grid
n_max = L
nx, ny = np.meshgrid(np.arange(0, n_max + 1), np.arange(0, n_max + 1))
n_sq = nx ** 2 + ny ** 2
unique_n_sq = np.unique(n_sq)
unique_n_sq = unique_n_sq[unique_n_sq > 0]  # Remove 0 to avoid division by zero
available_lambdas = np.sort(L / np.sqrt(unique_n_sq))

# Theoretical slope for the continuous model
m_continuous = (
        (2 * np.pi / np.sqrt(config["decay_rate_gamma"]))
        * ((np.sqrt(rho_0 / rho_c) - 1) ** (-0.5))
)


# 4. Model definitions
def continuous_model(d_chem):
    return m_continuous * np.sqrt(d_chem)


def discrete_model(d_chem, dx_val=dx):
    lam_cont = continuous_model(d_chem)
    arg = (np.pi * dx_val) / lam_cont
    with np.errstate(invalid='ignore'):
        lam_disc = np.where(arg <= 1.0, (np.pi * dx_val) / np.arcsin(arg), np.nan)
    return lam_disc


def empirical_linear_model(sqrt_d_chem, m, c):
    return m * sqrt_d_chem + c


# 5. Perform the Empirical Fit
popt, pcov = curve_fit(
    empirical_linear_model,
    sqrt_D_chem,
    lambda_max_exp,
    sigma=lambda_max_err,
    absolute_sigma=True
)
m_fit, C_fit = popt
m_fit_err, C_fit_err = np.sqrt(np.diag(pcov))
sigma_diff = abs(m_fit - m_continuous) / m_fit_err

# 6. Find the 1st, 2nd, and 3rd closest discrete 2D modes
closest_1_x, closest_1_y = [], []
closest_2_x, closest_2_y = [], []
closest_3_x, closest_3_y = [], []

for x_val in sqrt_D_chem:
    y_fit = empirical_linear_model(x_val, m_fit, C_fit)
    diffs = np.abs(available_lambdas - y_fit)

    # Get indices of the 3 smallest differences
    closest_idx = np.argsort(diffs)[:3]

    # Append to respective lists
    if len(closest_idx) > 0:
        closest_1_x.append(x_val)
        closest_1_y.append(available_lambdas[closest_idx[0]])
    if len(closest_idx) > 1:
        closest_2_x.append(x_val)
        closest_2_y.append(available_lambdas[closest_idx[1]])
    if len(closest_idx) > 2:
        closest_3_x.append(x_val)
        closest_3_y.append(available_lambdas[closest_idx[2]])

# 7. Reduced Chi-Squared & Residual Analysis
N = len(lambda_max_exp)
lambda_theo_cont = continuous_model(D_chem)
lambda_theo_fit = empirical_linear_model(sqrt_D_chem, m_fit, C_fit)

chi2_cont = np.sum(((lambda_max_exp - lambda_theo_cont) / lambda_max_err) ** 2)
red_chi2_cont = chi2_cont / N

chi2_fit = np.sum(((lambda_max_exp - lambda_theo_fit) / lambda_max_err) ** 2)
red_chi2_fit = chi2_fit / (N - 2)
res_fit = lambda_max_exp - lambda_theo_fit

# 8. Visualization
sqrt_D_smooth = np.linspace(min(sqrt_D_chem) * 0.9, max(sqrt_D_chem) * 1.05, 300)
D_chem_smooth = sqrt_D_smooth ** 2
lambda_smooth_cont = continuous_model(D_chem_smooth)
lambda_smooth_fit = empirical_linear_model(sqrt_D_smooth, m_fit, C_fit)

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(10, 8),
    gridspec_kw={'height_ratios': [3, 1]},
    sharex=True
)

# --- MAIN PLOT (ax1) ---
ax1.errorbar(
    sqrt_D_chem, lambda_max_exp, yerr=lambda_max_err,
    fmt='o', color='black', ecolor='gray', capsize=4,
    label='Simulation Data', markersize=6, zorder=4
)

ax1.plot(
    sqrt_D_smooth, lambda_smooth_cont, color='#555555', linestyle='--', linewidth=2,
    label=f'Continuous Theory ($m={m_continuous:.2f}$)', zorder=2
)

ax1.plot(
    sqrt_D_smooth, lambda_smooth_fit, 'b:', linewidth=2,
    label=f'Empirical Fit ($m={m_fit:.2f}, C={C_fit:.2f}$)', zorder=3
)

if INCLUDE_DISCRETE_MODEL:
    lambda_smooth_disc = discrete_model(D_chem_smooth)
    ax1.plot(
        sqrt_D_smooth, lambda_smooth_disc, 'g-', linewidth=2, alpha=0.5,
        label=f'Discrete Lattice Theory (1D correction)', zorder=1
    )

# Plot the nearest available 2D modes as ranked scatters
ax1.scatter(
    closest_3_x, closest_3_y,
    color='gold', marker='x', s=60, zorder=5, linewidths=2,
    label='3rd Closest 2D Mode'
)
ax1.scatter(
    closest_2_x, closest_2_y,
    color='darkorange', marker='x', s=70, zorder=6, linewidths=2,
    label='2nd Closest 2D Mode'
)
ax1.scatter(
    closest_1_x, closest_1_y,
    color='crimson', marker='x', s=80, zorder=7, linewidths=2.5,
    label='1st Closest 2D Mode'
)

title_str = r'Cluster Separation: $\lambda_{max}$ vs. $\sqrt{D_{chem}}$'
ax1.set_title(title_str, fontsize=14)
ax1.set_ylabel(r'$\lambda_{max}\quad (\mathrm{Cluster\ Separation})$', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='lower right', framealpha=0.9, fontsize=9)

# Summary statistics text box
stats_text = (
    f"Gradient Analysis:\n"
    f"  $m_{{fit}}$ = {m_fit:.2f} $\\pm$ {m_fit_err:.2f}\n"
    f"  $m_{{theory}}$ = {m_continuous:.2f}\n"
    f"  $\\Delta$ = {sigma_diff:.1f} $\\sigma$\n\n"
    f"$\\chi^2_\\nu$ Comparison:\n"
    f"  Fit (Linear): {red_chi2_fit:.2f}\n"
    f"  Theory (Cont.): {red_chi2_cont:.2f}"
)
props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
ax1.text(
    0.05, 0.95, stats_text, transform=ax1.transAxes,
    fontsize=11, verticalalignment='top', bbox=props
)

# --- RESIDUAL PLOT (ax2) ---
ax2.axhline(0, color='black', linewidth=1.5, linestyle='-', alpha=0.7)

ax2.errorbar(
    sqrt_D_chem, res_fit, yerr=lambda_max_err,
    fmt='s', color='blue', ecolor='lightblue', capsize=3, label='Fit Residuals'
)

ax2.set_xlabel(r'$\sqrt{D_{chem}}\quad (\mathrm{Diffusion\ Rate}^{1/2})$', fontsize=12)
ax2.set_ylabel('Residuals', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()