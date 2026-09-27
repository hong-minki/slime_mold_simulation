import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.optimize import minimize
import csv

def load_data(folder_path):
    config_path = folder_path / "sim_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
        
    cell_raw = np.fromfile(folder_path / "cell_history.bin", dtype=np.int32)
    return config, cell_raw

def get_max_density_series(cell_stack):
    return np.max(cell_stack, axis=(1, 2))
        
def compute_power_spectrum(grid):
    grid_fluctuation = grid - np.mean(grid)
    fft2d = np.fft.fft2(grid_fluctuation)
    power_spectrum = np.abs(fft2d)**2
    
    Ny, Nx = grid.shape
    fx = np.fft.fftfreq(Nx)
    fy = np.fft.fftfreq(Ny)
    fx_mesh, fy_mesh = np.meshgrid(fx, fy)
    f_mesh = np.sqrt(fx_mesh**2 + fy_mesh**2)
    
    return f_mesh.ravel(), power_spectrum.ravel()

def get_power_spectra_series(cell_stack):
    spectra_history = []
    f_array = None
    
    for grid in cell_stack:
        f_arr, spectrum = compute_power_spectrum(grid)
        if f_array is None:
            f_array = f_arr
        spectra_history.append(spectrum)
        
    return f_array, np.array(spectra_history)

def bin_spectra_series(f_array, spectra_series, bin_width=None):
    if bin_width is None:
        bin_width = np.min(f_array[f_array > 0])
    
    f_binned = np.floor(f_array / bin_width).astype(int)
    max_bin = np.max(f_binned)
    nr = np.bincount(f_binned, minlength=max_bin + 1)
    
    valid_bins = nr > 0
    enough_points = nr >= 3
    binned_f = (np.arange(max_bin + 1) + 0.5) * bin_width
    
    binned_spectra_series = []
    binned_sem_series = []
    
    for spectrum in spectra_series:
        tbin = np.bincount(f_binned, weights=spectrum, minlength=max_bin + 1)
        radial_profile = np.zeros(max_bin + 1, dtype=np.float64)
        radial_profile[valid_bins] = tbin[valid_bins] / nr[valid_bins]
        
        tbin_sq = np.bincount(f_binned, weights=spectrum**2, minlength=max_bin + 1)
        mean_sq_power = np.zeros(max_bin + 1, dtype=np.float64)
        mean_sq_power[valid_bins] = tbin_sq[valid_bins] / nr[valid_bins]
        
        variance = np.maximum(mean_sq_power - radial_profile**2, 0)
        sem = np.zeros(max_bin + 1, dtype=np.float64)
        
        sem[enough_points] = np.sqrt(variance[enough_points]) / np.sqrt(nr[enough_points]) + 1e-10
        sem[~enough_points] = np.inf
        
        binned_spectra_series.append(radial_profile)
        binned_sem_series.append(sem)
        
    return binned_f, np.array(binned_spectra_series), np.array(binned_sem_series)

def gaussian_model(B, x):
    return B[0] * np.exp(-((x - B[1])**2) / (2 * B[2]**2)) + B[3]

def extract_f_max_unbinned_mle(f_array, p_raw, window_center, window_radius_left, window_radius_right):
    mask = (f_array >= max(1e-5, window_center - window_radius_left)) & (f_array <= (window_center + window_radius_right))
    f_fit = f_array[mask]
    p_fit = p_raw[mask]
    
    if len(f_fit) < 10:
        print("Warning: Not enough unbinned pixels in window to run MLE.")
        return window_center, 0.0, None

    p_scale = np.mean(p_fit)
    p_norm = p_fit / p_scale

    def nll_unbinned(params):
        amp, mean, stddev, offset = params
        if amp <= 0 or stddev <= 0 or offset <= 0:
            return np.inf
        S = gaussian_model(params, f_fit)
        return np.sum(np.log(S) + (p_norm / S))

    initial_guess = [2.0, window_center, (window_radius_left + window_radius_right) / 4.0, 0.5]
    bounds = [(1e-5, np.inf), (f_fit.min(), f_fit.max()), (1e-5, np.inf), (1e-5, np.inf)]
    
    result = minimize(nll_unbinned, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    if result.success:
        f_max_continuous = result.x[1]
        amp, mean, std, offset = result.x
        S_opt = gaussian_model(result.x, f_fit)
        
        exp_term = np.exp(-((f_fit - mean)**2) / (2 * std**2))
        J = np.column_stack([
            exp_term, 
            amp * exp_term * (f_fit - mean) / (std**2), 
            amp * exp_term * ((f_fit - mean)**2) / (std**3), 
            np.ones_like(f_fit)
        ])
        
        FIM = J.T @ ((1.0 / (S_opt**2))[:, None] * J)
        
        try:
            f_max_error = np.sqrt(np.linalg.inv(FIM)[1, 1])
        except np.linalg.LinAlgError:
            f_max_error = 0.0
            
        result.x[0] *= p_scale 
        result.x[3] *= p_scale 
        return f_max_continuous, f_max_error, result.x
    
    initial_guess[0] *= p_scale
    initial_guess[3] *= p_scale
    return window_center, 0.0, initial_guess

# --- UPDATED FUNCTION SIGNATURE AND TITLE ---
def plot_mle_overlap(f_raw, p_raw, f_binned, p_binned, p_err_binned, popt, f_max, f_max_error, window_center, window_radius_left, window_radius_right, d_chem_val):
    if popt is None:
        return

    mask_raw = (f_raw >= max(1e-5, window_center - window_radius_left)) & (f_raw <= (window_center + window_radius_right))
    f_fit, p_fit = f_raw[mask_raw], p_raw[mask_raw]
    
    mask_binned = (f_binned >= f_fit.min()) & (f_binned <= f_fit.max())
    f_bin_fit, p_bin_fit, p_err_fit = f_binned[mask_binned], p_binned[mask_binned], p_err_binned[mask_binned]

    plt.figure(figsize=(8, 5))
    plt.scatter(f_fit, p_fit, color='dodgerblue', s=5, alpha=0.15, label='Raw Unbinned Pixels', zorder=1)
    plt.errorbar(f_bin_fit, p_bin_fit, yerr=p_err_fit, fmt='o', color='black', markersize=5, capsize=3, label='Binned Data (SEM)', zorder=2)
                 
    x_smooth = np.linspace(f_fit.min(), f_fit.max(), 200)
    plt.plot(x_smooth, gaussian_model(popt, x_smooth), color='crimson', lw=2.5, label=f'MLE Fit\n$f_{{max}} = {f_max:.4f} \pm {f_max_error:.4f}$', zorder=3)
    plt.axvline(f_max, color='crimson', linestyle='--', alpha=0.8, zorder=3)
    
    plt.title(f'True Unbinned Maximum Likelihood Fit vs Binned Average ($D_{{chem}} = {d_chem_val}$)')
    plt.xlabel('Spatial Frequency $f$ ($1/\lambda$)')
    plt.ylabel('Power')
    plt.ylim(0, max(np.max(gaussian_model(popt, x_smooth)), np.max(p_bin_fit + p_err_fit)) * 1.3)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def process_ensemble_for_dchem(d_chem_str, d_chem_val, results_dir, step_skip=10, plot_mle=False):
    # Use the EXACT string from the folder name for the pattern
    pattern = f"Simulation*_new_Dchem_{d_chem_str}"
    folder_paths = sorted(list(results_dir.glob(pattern)))

    if not folder_paths:
        print(f"[!] Warning: No folders found matching {pattern}.")
        return None

    print(f"Processing D_chem = {d_chem_str} | Ensembles found: {len(folder_paths)}")

    all_max_densities, all_spectra_series = [], []
    f_array_single, config = None, None

    for folder_path in folder_paths:
        c, cell_raw = load_data(folder_path)
        if config is None:
            config = c
            
        c_stack = cell_raw.reshape(-1, config["height"], config["width"])[::step_skip]
        all_max_densities.append(get_max_density_series(c_stack))
        
        f_arr, spec_series = get_power_spectra_series(c_stack)
        all_spectra_series.append(spec_series)
        
        if f_array_single is None:
            f_array_single = f_arr

    min_frames = min(len(dens) for dens in all_max_densities)
    all_max_densities = [dens[:min_frames] for dens in all_max_densities]
    all_spectra_series = [spec[:min_frames, :] for spec in all_spectra_series]

    mean_max_density_series = np.mean(all_max_densities, axis=0)
    combined_f_array = np.tile(f_array_single, len(folder_paths))
    combined_spectra_series = np.concatenate(all_spectra_series, axis=1)

    bin_width = 1 / config["width"]
    binned_f, binned_spectra, binned_sem = bin_spectra_series(combined_f_array, combined_spectra_series, bin_width=bin_width)

    threshold = (config["total_cells"] / (config["width"] * config["height"])) * 1.1
    valid_indices = np.where(mean_max_density_series < threshold)[0]
    best_frame_idx = valid_indices[-1] if len(valid_indices) > 0 else 0

    best_power_unbinned = combined_spectra_series[best_frame_idx]
    best_power_binned = binned_spectra[best_frame_idx]
    best_sem_binned = binned_sem[best_frame_idx]

    valid_mask = (binned_f > 0) & (binned_f <= 0.5)
    discrete_peak_idx = np.argmax(best_power_binned[valid_mask])
    window_center = binned_f[valid_mask][discrete_peak_idx]

    window_radius = bin_width * 6
    f_max, f_error, popt = extract_f_max_unbinned_mle(
        combined_f_array, best_power_unbinned, window_center, window_radius, window_radius
    )

    lambda_max = 1.0 / f_max if f_max > 0 else 0.0
    lambda_max_err = (f_error / f_max) * lambda_max if f_max > 0 else 0.0

    # --- UPDATED FUNCTION CALL ---
    if plot_mle:
        plot_mle_overlap(
            combined_f_array, best_power_unbinned, binned_f, best_power_binned, best_sem_binned,
            popt, f_max, f_error, window_center, window_radius, window_radius, d_chem_val
        )

    # Return the parsed float value for the CSV
    return [d_chem_val, f_max, f_error, lambda_max, lambda_max_err]


# ==========================================
# GLOBAL EXECUTION 
# ==========================================
if __name__ == "__main__":
    PLOT_MLE = True
    STEP_SKIP = 10
    CSV_FILENAME = "cluster_separation_data.csv"
    
    script_dir = Path(__file__).resolve().parent
    results_dir = (script_dir / ".." / ".." / "results").resolve()
    
    unique_d_chems = set()
    
    if results_dir.exists():
        for folder in results_dir.glob("Simulation*_new_Dchem_*"):
            if folder.is_dir():
                try:
                    # Extract the exact string right after 'Dchem_' (e.g., '32.0' or '18.53')
                    val_str = folder.name.split("Dchem_")[-1]
                    # We only add it if it can successfully cast to float
                    float(val_str)
                    unique_d_chems.add(val_str)
                except (IndexError, ValueError):
                    continue
    else:
        print(f"[!] Results directory not found: {results_dir}")
        
    # Sort them in numerical ascending order by casting to float temporarily during sorting
    D_chem_list_str = sorted(list(unique_d_chems), key=float)
    
    if not D_chem_list_str:
        print("No valid D_chem folders found. Please check your results directory.")
    else:
        print(f"Found {len(D_chem_list_str)} unique D_chem values: {D_chem_list_str}")

    csv_data = []

    for d_chem_str in D_chem_list_str:
        # Keep the exact string for the folder search, but pass the float value for the math/CSV
        d_chem_val = float(d_chem_str)
        result_row = process_ensemble_for_dchem(d_chem_str, d_chem_val, results_dir, step_skip=STEP_SKIP, plot_mle=PLOT_MLE)
        
        if result_row:
            csv_data.append(result_row)

    if csv_data:
        with open(CSV_FILENAME, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["D_chem", "f_max", "f_max_err", "lambda_max", "lambda_max_err"])
            writer.writerows(csv_data)

        print(f"\nSUCCESS! Batch processing complete. Data saved to {CSV_FILENAME}")