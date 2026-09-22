import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- TOGGLE FOR DISCRETE MODEL ANALYSIS ---
INCLUDE_DISCRETE_MODEL = False  # Set to False to exclude the discrete model
# ------------------------------------------

# 1. Load data from CSV
df = pd.read_csv('cluster_separation_data.csv')

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
rho_0 = config["total_cells"] / (config["width"] * config["height"])
rho_c = (config["Dr"] * config["decay_rate_gamma"]) / (config["chi"] * config["chem_secretion_rate_alpha"])

# Theoretical slope for the continuous model: lambda_continuous = m * sqrt(D_chem)
m_continuous = (
    (2 * np.pi / np.sqrt(config["decay_rate_gamma"]))
    * ((np.sqrt(rho_0 / rho_c) - 1) ** (-0.5))
)

# 3. Model definitions
def continuous_model(d_chem):
    """Calculates continuous theoretical wavelength lambda_continuous."""
    return m_continuous * np.sqrt(d_chem)

def discrete_model(d_chem, dx_val=dx):
    """
    Calculates discrete lattice-corrected theoretical wavelength:
    lambda_discrete = (pi * dx) / arcsin(pi * dx / lambda_continuous)
    """
    lam_cont = continuous_model(d_chem)
    arg = (np.pi * dx_val) / lam_cont
    
    # Clip/handle values exceeding the Nyquist catastrophe domain (arg > 1)
    with np.errstate(invalid='ignore'):
        lam_disc = np.where(arg <= 1.0, (np.pi * dx_val) / np.arcsin(arg), np.nan)
    return lam_disc

# Values at experimental data points
lambda_theo_cont = continuous_model(D_chem)

# 4. Reduced Chi-Squared Analysis (Degrees of freedom = N, p = 0)
dof = len(lambda_max_exp)

chi2_cont = np.sum(((lambda_max_exp - lambda_theo_cont) / lambda_max_err) ** 2)
red_chi2_cont = chi2_cont / dof

print("--- Goodness-of-Fit Comparison ---")
print(f"Degrees of freedom (N): {dof}")
print(f"Continuous Model: Chi^2 = {chi2_cont:.4f}, Reduced Chi^2 = {red_chi2_cont:.4f}")

if INCLUDE_DISCRETE_MODEL:
    lambda_theo_disc = discrete_model(D_chem)
    chi2_disc = np.sum(((lambda_max_exp - lambda_theo_disc) / lambda_max_err) ** 2)
    red_chi2_disc = chi2_disc / dof
    print(f"Discrete Model:   Chi^2 = {chi2_disc:.4f}, Reduced Chi^2 = {red_chi2_disc:.4f}")

# 5. Curve generation for plotting
sqrt_D_smooth = np.linspace(min(sqrt_D_chem), max(sqrt_D_chem), 300)
D_chem_smooth = sqrt_D_smooth ** 2

lambda_smooth_cont = continuous_model(D_chem_smooth)

# 6. Visualization
plt.figure(figsize=(10, 6))

# Experimental data
plt.errorbar(
    sqrt_D_chem, 
    lambda_max_exp, 
    yerr=lambda_max_err, 
    fmt='o', 
    color='tab:blue', 
    ecolor='black', 
    capsize=4, 
    label='Simulation Data', 
    markersize=6,
    zorder=3
)

# Continuous linear model
plt.plot(
    sqrt_D_smooth, 
    lambda_smooth_cont, 
    'r--', 
    linewidth=2, 
    label=f'Continuous Theory ($\\chi^2_\\nu = {red_chi2_cont:.2f}$)'
)

# Discrete lattice-corrected model (conditional)
if INCLUDE_DISCRETE_MODEL:
    lambda_smooth_disc = discrete_model(D_chem_smooth)
    plt.plot(
        sqrt_D_smooth, 
        lambda_smooth_disc, 
        'g-', 
        linewidth=2, 
        label=f'Discrete Lattice Theory ($\\chi^2_\\nu = {red_chi2_disc:.2f}$)'
    )

title_str = r'Cluster Separation: $\lambda_{max}$ vs. $\sqrt{D_{chem}}$'
if INCLUDE_DISCRETE_MODEL:
    title_str += ' (Continuous vs. Discrete Grid)'
plt.title(title_str)
plt.xlabel(r'$\sqrt{D_{chem}}\quad (\mathrm{Diffusion\ Rate}^{1/2})$')
plt.ylabel(r'$\lambda_{max}\quad (\mathrm{Cluster\ Separation})$')
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right', framealpha=0.9)

# Summary statistics text box
stats_text = (
    f"Continuous Model:\n"
    f"  $\\chi^2_\\nu = {red_chi2_cont:.2f}$"
)
if INCLUDE_DISCRETE_MODEL:
    stats_text += (
        f"\nDiscrete Model:\n"
        f"  $\\chi^2_\\nu = {red_chi2_disc:.2f}$"
    )

props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
plt.gca().text(
    0.05, 0.95, 
    stats_text, 
    transform=plt.gca().transAxes, 
    fontsize=11,
    verticalalignment='top', 
    bbox=props
)

plt.tight_layout()
plt.show()