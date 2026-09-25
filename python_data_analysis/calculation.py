import json
from pathlib import Path
import numpy as np

def load_config(folder_name):
    script_dir = Path(__file__).resolve().parent
    warehouse_path = (script_dir / ".." / ".." / "results" / folder_name).resolve()
    
    config_path = warehouse_path / "sim_config.json"
    
    # Fallback for testing the script locally if the path doesn't exist yet
    if not config_path.exists():
        print(f"Warning: {config_path} not found. Using default parameters.")
        return {
            "width": 32, "height": 32, "total_cells": 10240000, 
            "diffusion_rate": 16.0, "Dr": 0.5, "decay_rate": 0.1, 
            "chem_secretion_rate": 1.0, "chi": 2.0
        }

    with open(config_path, "r") as f:
        config = json.load(f)
    
    return config

def get_params(config):
    """Helper function to extract and calculate base parameters cleanly."""
    width = config["width"]
    height = config["height"]
    total_cells = config["total_cells"]
    
    rho_0 = total_cells / (width * height)
    D_ch = config["diffusion_rate"]
    D_ce = config["Dr"]
    gamma = config["decay_rate"]
    alpha = config["chem_secretion_rate"]
    chi = config["chi"]
    
    rho_c = (D_ce * gamma) / (chi * alpha)
    
    return rho_0, D_ch, D_ce, gamma, alpha, chi, rho_c

def calc_k_max_sq(config):
    rho_0, D_ch, _, gamma, _, _, rho_c = get_params(config)
    return (gamma / D_ch) * (np.sqrt(rho_0 / rho_c) - 1)

def dispersion_relation(k_sq, config):
    rho_0, D_ch, D_ce, gamma, alpha, chi, _ = get_params(config)
    omega = -D_ce * k_sq + (chi * rho_0 * alpha * k_sq) / (gamma + D_ch * k_sq)
    return omega

def chem_relaxation(k_sq, config):
    _, D_ch, _, gamma, _, _, _ = get_params(config)
    return gamma + D_ch * k_sq

if __name__ == "__main__":
    FOLDER_NAME = "Simulation1_Dchem_16.0"
    config = load_config(FOLDER_NAME)

    # 1. Calculate values
    k_max_sq_val = calc_k_max_sq(config)
    omega_max = dispersion_relation(k_max_sq_val, config)
    gamma_c = chem_relaxation(k_max_sq_val, config)
    
    # 2. Print results
    print(f"--- Analysis for {FOLDER_NAME} ---")
    print(f"D_chem: {config['diffusion_rate']}")
    print(f"k_max^2: {k_max_sq_val:.4f}")
    print("-" * 35)
    print(f"Cell Clustering Rate (omega_max): {omega_max:.2f}")
    print(f"Chemical Relaxation Rate (Gamma_c): {gamma_c:.2f}")
    print("-" * 35)
    
    # 3. Evaluate the assumption
    ratio = gamma_c / omega_max
    print(f"Ratio (Gamma_c / omega_max): {ratio:.4f}")
    
    if ratio > 10.0:
        print("Conclusion: Quasi-steady state assumption HOLDS (Gamma_c >> omega_max).")
    else:
        print("Conclusion: Quasi-steady state assumption FAILS. The chemical field is too slow.")
        
        # Calculate the required multiplier
        required_multiplier = 10.0 / ratio
        print(f"-> To fix this, multiply dt_chem by at least: {required_multiplier:.1f}x")