import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import imageio_ffmpeg
import json
import scipy.ndimage as ndi
from scipy.optimize import minimize
import glob

#run this first for animation %matplotlib qt

# Map the ffmpeg executable for saving videos
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

def load_data(folder_path):
    """Modified to take a Path object directly"""
    config_path = folder_path / "sim_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
        
    cell_raw = np.fromfile(folder_path / "cell_history.bin", dtype=np.int32)
    return config, cell_raw

def get_smoothed_cell_stack(cell_stack, method='box', size=3):
    smoothed_stack = []
    
    for grid in cell_stack:
        grid_float = grid.astype(np.float64)
        
        if method == 'box':
            smoothed_grid = ndi.uniform_filter(grid_float, size=size, mode='wrap')
        elif method == 'gaussian':
            smoothed_grid = ndi.gaussian_filter(grid_float, sigma=size, mode='wrap')
        else:
            raise ValueError("Method must be 'box' or 'gaussian'")
            
        smoothed_stack.append(smoothed_grid)
        
    return np.array(smoothed_stack)

def get_max_density_series(smoothed_cell_stack):
    return np.max(smoothed_cell_stack, axis=(1, 2))
        
def compute_power_spectrum(grid):
    grid_fluctuation = grid - np.mean(grid)
    fft2d = np.fft.fft2(grid_fluctuation)
    power_spectrum = np.abs(fft2d)**2
    power_spectrum_array = power_spectrum.ravel()
    
    Ny, Nx = grid.shape
    fx = np.fft.fftfreq(Nx)
    fy = np.fft.fftfreq(Ny)
    fx_mesh, fy_mesh = np.meshgrid(fx, fy)
    f_mesh = np.sqrt(fx_mesh**2 + fy_mesh**2)
    f_array = f_mesh.ravel()
    
    return f_array, power_spectrum_array

def get_power_spectra_series(cell_stack):
    spectra_history = []
    f_array = None
    
    for grid in cell_stack:
        f_arr, spectrum = compute_power_spectrum(grid)
        
        # Capture the k_array once (it's identical for all frames)
        if f_array is None:
            f_array = f_arr
            
        spectra_history.append(spectrum)
        
    return f_array, np.array(spectra_history)

def bin_spectra_series(f_array, spectra_series, bin_width=None):
    """
    Bins flattened 1D power spectra arrays into radially averaged frequency bins.
    
    Parameters:
    - f_array: 1D array of frequency magnitudes.
    - spectra_series: 2D array of shape (num_frames, len(f_array)) with raw power data.
    - bin_width: The width of each frequency bin. If None, it defaults to roughly 
                 one frequency step (1.0 / Nx).
    
    Returns:
    - binned_f: 1D array of bin centers.
    - binned_spectra_series: 2D array of averaged power per bin over time.
    - binned_sem_series: 2D array of Standard Error of the Mean per bin over time.
    """
    min_points = 3
    epsilon = 1e-10
    
    # If bin_width is not provided, estimate a good default (1 cycle per box width)
    # This assumes the minimum non-zero frequency step in f_array is roughly the bin_width
    if bin_width is None:
        bin_width = np.min(f_array[f_array > 0])
    
    # 1. Determine bin indices for each point in the raw f_array
    f_binned = np.floor(f_array / bin_width).astype(int)
    max_bin = np.max(f_binned)
    
    # 2. Count the number of points that fall into each bin
    # We use minlength to ensure arrays remain aligned up to max_bin
    nr = np.bincount(f_binned, minlength=max_bin + 1)
    
    valid_bins = nr > 0
    enough_points = nr >= min_points
    
    # 3. Create the binned frequency axis (using bin centers)
    binned_f = (np.arange(max_bin + 1) + 0.5) * bin_width
    
    # Pre-allocate output lists
    binned_spectra_series = []
    binned_sem_series = []
    
    # 4. Iterate over each frame and bin the power data
    for spectrum in spectra_series:
        # Sum of power per bin
        tbin = np.bincount(f_binned, weights=spectrum, minlength=max_bin + 1)
        radial_profile = np.zeros(max_bin + 1, dtype=np.float64)
        radial_profile[valid_bins] = tbin[valid_bins] / nr[valid_bins]
        
        # Sum of squared power per bin (for variance calculation)
        tbin_sq = np.bincount(f_binned, weights=spectrum**2, minlength=max_bin + 1)
        mean_sq_power = np.zeros(max_bin + 1, dtype=np.float64)
        mean_sq_power[valid_bins] = tbin_sq[valid_bins] / nr[valid_bins]
        
        # Calculate Variance and SEM
        variance = np.maximum(mean_sq_power - radial_profile**2, 0)
        sem = np.zeros(max_bin + 1, dtype=np.float64)
        
        sem[enough_points] = np.sqrt(variance[enough_points]) / np.sqrt(nr[enough_points]) + epsilon
        sem[~enough_points] = np.inf  # Mark bins with insufficient points
        
        binned_spectra_series.append(radial_profile)
        binned_sem_series.append(sem)
        
    return binned_f, np.array(binned_spectra_series), np.array(binned_sem_series)

#Fitting
def gaussian_model(B, x):
    """Gaussian peak: B[0]=Amplitude, B[1]=Mean(f_max), B[2]=StdDev, B[3]=Offset"""
    return B[0] * np.exp(-((x - B[1])**2) / (2 * B[2]**2)) + B[3]

def extract_f_max_unbinned_mle(f_array, p_raw, window_center, window_radius_left, window_radius_right):
    """
    Fits the theoretical Gaussian directly to the raw, unbinned Fourier pixels
    using a normalized Exponential Maximum Likelihood scale with an asymmetric window.
    """
    # Updated to use distinct left and right radius parameters
    mask = (f_array >= max(1e-5, window_center - window_radius_left)) & (f_array <= (window_center + window_radius_right))
    
    f_fit = f_array[mask]
    p_fit = p_raw[mask]
    print(f"total number of data is {len(f_fit)}")
    if len(f_fit) < 10:
        print("Warning: Not enough unbinned pixels in window to run MLE.")
        return window_center, 0.0, None

    # Normalize the power array so the optimizer doesn't choke on 10^10 scale values
    p_scale = np.mean(p_fit)
    p_norm = p_fit / p_scale

    def nll_unbinned(params):
        amp, mean, stddev, offset = params
        if amp <= 0 or stddev <= 0 or offset <= 0:
            return np.inf
        # S is now the expected normalized power
        S = gaussian_model(params, f_fit)
        return np.sum(np.log(S) + (p_norm / S))

    # Initial guesses scaled to the normalized data (values around ~1.0)
    amp_guess = 2.0  
    mean_guess = window_center
    # Update standard deviation guess to be the average of the two radii
    stddev_guess = (window_radius_left + window_radius_right) / 4.0 
    offset_guess = 0.5 
    
    initial_guess = [amp_guess, mean_guess, stddev_guess, offset_guess]
    
    bounds = [
        (1e-5, np.inf),             
        (f_fit.min(), f_fit.max()), 
        (1e-5, np.inf),             
        (1e-5, np.inf)              
    ]
    
    result = minimize(nll_unbinned, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    if result.success:
        f_max_continuous = result.x[1]
        
        # --- CALCULATE TRUE STATISTICAL ERROR (Fisher Information) ---
        amp, mean, std, offset = result.x
        S_opt = gaussian_model(result.x, f_fit)
        
        # Partial derivatives of the model with respect to each parameter
        exp_term = np.exp(-((f_fit - mean)**2) / (2 * std**2))
        
        dS_dA = exp_term
        dS_dmean = amp * exp_term * (f_fit - mean) / (std**2)
        dS_dstd = amp * exp_term * ((f_fit - mean)**2) / (std**3)
        dS_doffset = np.ones_like(f_fit)
        
        # Jacobian matrix (N_points x 4_parameters)
        J = np.column_stack([dS_dA, dS_dmean, dS_dstd, dS_doffset])
        
        W = 1.0 / (S_opt**2)
        FIM = J.T @ (W[:, None] * J)
        
        # True Covariance Matrix is the inverse of Fisher Info
        try:
            true_cov = np.linalg.inv(FIM)
            f_max_error = np.sqrt(true_cov[1, 1])
        except np.linalg.LinAlgError:
            print("Warning: Fisher Matrix singular, falling back to 0 error.")
            f_max_error = 0.0
            
        # Rescale the Amplitude and Offset back to physical units for plotting
        result.x[0] *= p_scale 
        result.x[3] *= p_scale 
        
        print("-> Unbinned MLE Optimization Succeeded.")
        print(f"f_max = {f_max_continuous:.5f} ± {f_max_error:.5f}")
        return f_max_continuous, f_max_error, result.x
    else:
        print("\n[!] Unbinned MLE Failed to converge. Resorting to discrete guess.")
        # Scale back the initial guess just so it plots something visible
        initial_guess[0] *= p_scale
        initial_guess[3] *= p_scale
        return mean_guess, 0.0, initial_guess
#Visualisation
def plot_max_density(max_density_series, dt, step_skip, rho_0=None):
    time_steps = np.arange(len(max_density_series)) * dt
    
    plt.figure(figsize=(8, 5))
    plt.plot(time_steps * step_skip, max_density_series, color='royalblue', lw=2, label=r'$\max(\rho)$')
    
    if rho_0 is not None:
        plt.axhline(1.1 * rho_0, color='crimson', linestyle='--', label=r'10% Threshold ($1.1\rho_0$)')
        plt.axhline(rho_0, color='gray', linestyle=':', label=r'Mean Density ($\rho_0$)')
        
    plt.xlabel('Time (time_steps * dt)')
    plt.ylabel('Maximum Local Density')
    plt.title('Evolution of Peak Density')
    plt.grid(True, linestyle='--', alpha=0.6)
    if rho_0 is not None:
        plt.legend()
    plt.tight_layout()
    plt.show()

def animate_combined(f_array, smoothed_stack, cell_stack, spectra_series, sem_series, dt, step_skip, interval):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    smooth_vmin, smooth_vmax = smoothed_stack.min(), smoothed_stack.max()
    img_smooth = ax1.imshow(
        smoothed_stack[0], cmap="inferno", vmin=smooth_vmin, vmax=smooth_vmax, origin="lower"
    )
    ax1.set_title("Smoothed Cell Density", fontsize=13)
    fig.colorbar(img_smooth, ax=ax1, fraction=0.046, pad=0.04, label="Density (smoothed)")
    ax1.set_xlabel("x coordinate")
    ax1.set_ylabel("y coordinate")

    cell_vmin, cell_vmax = cell_stack.min(), cell_stack.max()
    img_cell = ax2.imshow(
        cell_stack[0], cmap="inferno", vmin=cell_vmin, vmax=cell_vmax, origin="lower"
    )
    ax2.set_title("Raw Cell Distribution", fontsize=13)
    fig.colorbar(img_cell, ax=ax2, fraction=0.046, pad=0.04, label="Cell Count")
    ax2.set_xlabel("x coordinate")
    ax2.set_ylabel("y coordinate")

    num_bins = spectra_series.shape[1]
    global_max = np.max(spectra_series[:, 1:])
    

    # Scale X axis limits using the actual f_array bounds
    ax3.set_xlim(f_array[1], f_array[-1])
    ax3.set_ylim(0, global_max * 1.3)
    ax3.set_title("Evolution of Power Spectrum", fontsize=13)
    ax3.set_xlabel("Spatial Frequency $f$ ($1/\lambda$)")
    ax3.set_ylabel("Averaged Power")
    ax3.grid(True, linestyle="--", alpha=0.6)

    line, caplines, barlinecols = ax3.errorbar(
        f_array, np.zeros(num_bins), yerr=np.zeros(num_bins), 
        fmt='-', lw=2.5, color="dodgerblue", elinewidth=1.5, alpha=0.8
    )
    error_lines = barlinecols[0] 

    time_text = ax3.text(
        0.55, 0.90, "", transform=ax3.transAxes, fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
    )

    plt.tight_layout()

    def init():
        img_smooth.set_data(smoothed_stack[0])
        img_cell.set_data(cell_stack[0])
        line.set_data([], [])
        
        empty_segments = [np.array([[x, 0], [x, 0]]) for x in f_array]
        error_lines.set_segments(empty_segments)
        
        time_text.set_text("")
        return img_smooth, img_cell, line, error_lines, time_text

    def update(frame):
        img_smooth.set_data(smoothed_stack[frame])
        img_cell.set_data(cell_stack[frame])
        
        y = spectra_series[frame]
        err = sem_series[frame].copy()
        
        err[np.isinf(err)] = np.nan
        
        # Plot against f_array
        line.set_data(f_array, y)
        
        segments = [np.array([[x, y_val - e], [x, y_val + e]]) for x, y_val, e in zip(f_array, y, err)]
        error_lines.set_segments(segments)
        
        time_text.set_text(f"Time: {frame * step_skip * dt:.2f}")
        
        return img_smooth, img_cell, line, error_lines, time_text

    ani = FuncAnimation(
        fig, update, frames=len(cell_stack), init_func=init, blit=True, interval=interval
    )

    return ani

def plot_mle_overlap(f_raw, p_raw, f_binned, p_binned, p_err_binned, popt, f_max, f_max_error, window_center, window_radius_left, window_radius_right):
    """
    Plots the raw unbinned pixels, the binned empirical data with error bars,
    and the theoretical MLE Gaussian fit using an asymmetric window.
    """
    if popt is None:
        print("No valid fit parameters provided to plot.")
        return

    # Isolate the exact window used for the fit with the new left/right parameters
    mask_raw = (f_raw >= max(1e-5, window_center - window_radius_left)) & (f_raw <= (window_center + window_radius_right))
    f_fit = f_raw[mask_raw]
    p_fit = p_raw[mask_raw]
    
    # Isolate the binned points that fall inside the window
    mask_binned = (f_binned >= f_fit.min()) & (f_binned <= f_fit.max())
    f_bin_fit = f_binned[mask_binned]
    p_bin_fit = p_binned[mask_binned]
    p_err_fit = p_err_binned[mask_binned]

    plt.figure(figsize=(8, 5))
    
    # 1. Background: Raw unbinned pixels
    plt.scatter(f_fit, p_fit, color='dodgerblue', s=5, alpha=0.15, label='Raw Unbinned Pixels', zorder=1)
    
    # 2. Middleground: Binned data with standard error
    plt.errorbar(f_bin_fit, p_bin_fit, yerr=p_err_fit, fmt='o', color='black', 
                 markersize=5, capsize=3, label='Binned Data (SEM)', zorder=2)
                 
    # 3. Foreground: The MLE Gaussian Fit
    x_smooth = np.linspace(f_fit.min(), f_fit.max(), 200)
    y_smooth = gaussian_model(popt, x_smooth)
    
    plt.plot(x_smooth, y_smooth, color='crimson', lw=2.5, 
             label=f'MLE Fit\n$f_{{max}} = {f_max:.4f} \pm {f_max_error:.4f}$', zorder=3)
    
    plt.axvline(f_max, color='crimson', linestyle='--', alpha=0.8, zorder=3)
    
    plt.title('True Unbinned Maximum Likelihood Fit vs Binned Average')
    plt.xlabel('Spatial Frequency $f$ ($1/\lambda$)')
    plt.ylabel('Power')
    
    # Scale Y-axis based on the binned points to prevent massive raw outliers from compressing the plot
    max_y = max(np.max(y_smooth), np.max(p_bin_fit + p_err_fit))
    plt.ylim(0, max_y * 1.3)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    
# ==========================================
# GLOBAL EXECUTION 
# ==========================================
PATTERN = "Simulation*_Dchem_64.0" 

script_dir = Path(__file__).resolve().parent
results_dir = (script_dir / ".." / ".." / "results").resolve()
folder_paths = sorted(list(results_dir.glob(PATTERN)))

if not folder_paths:
    raise ValueError(f"No folders found matching {PATTERN} in {results_dir}")

print(f"Found {len(folder_paths)} simulation(s) for ensemble averaging. Loading data...")

all_max_densities = []
all_spectra_series = []

rep_cell_stack = None
rep_smoothed_stack = None
f_array_single = None
config = None

step_skip = 10
SAVE_VIDEO = False

# 1. Load and process all simulation folders
for i, folder_path in enumerate(folder_paths):
    c, cell_raw = load_data(folder_path)
    
    if config is None:
        config = c
        width = config["width"]
        height = config["height"]
        total_cells = config["total_cells"]
        dt = config["dt"]
        mean_cell_density = total_cells / (width * height)
        
    c_stack = cell_raw.reshape(-1, height, width)[::step_skip]
    
    all_max_densities.append(get_max_density_series(c_stack))
    
    f_arr, spec_series = get_power_spectra_series(c_stack)
    all_spectra_series.append(spec_series)
    
    if f_array_single is None:
        f_array_single = f_arr
        
    if i == 0:
        rep_cell_stack = c_stack
        rep_smoothed_stack = get_smoothed_cell_stack(c_stack, method='box', size=3)

# --- NEW ALIGNMENT STEP ---
# Find the shortest simulation length to safely align the time axis
min_frames = min([len(dens) for dens in all_max_densities])
print(f"Aligning simulations to minimum common frame count: {min_frames}")

# Crop all time-series lists to min_frames
all_max_densities = [dens[:min_frames] for dens in all_max_densities]
all_spectra_series = [spec[:min_frames, :] for spec in all_spectra_series]
rep_cell_stack = rep_cell_stack[:min_frames]
rep_smoothed_stack = rep_smoothed_stack[:min_frames]


# 2. Combine Ensemble Data Safely
mean_max_density_series = np.mean(all_max_densities, axis=0)

num_sims = len(folder_paths)
combined_f_array = np.tile(f_array_single, num_sims)

# Concatenate power spectra side-by-side. 
combined_spectra_series = np.concatenate(all_spectra_series, axis=1)

print("Binning ensemble power spectra...")
bin_width = 1 / width
binned_f, binned_spectra, binned_sem = bin_spectra_series(
    combined_f_array, 
    combined_spectra_series, 
    bin_width=bin_width 
)
window_radius_left = bin_width * 8
window_radius_right = bin_width * 6

# --- FIND THE BEST FRAME (Based on Ensemble Mean) ---
threshold = mean_cell_density * 1.1
valid_indices = np.where(mean_max_density_series < threshold)[0]

if len(valid_indices) == 0:
    print("Warning: Entire ensemble simulation exceeded the 10% non-linear threshold.")
    best_frame_idx = 0
else:
    best_frame_idx = valid_indices[-1]
    
print(f"\nBest Linear Frame Identified: Index {best_frame_idx} (Time: {best_frame_idx * step_skip * dt:.3f})")

best_power_unbinned = combined_spectra_series[best_frame_idx]
best_power_binned = binned_spectra[best_frame_idx]
best_sem_binned = binned_sem[best_frame_idx]

valid_mask = (binned_f > 0) & (binned_f <= 0.5)
valid_binned_f = binned_f[valid_mask]
valid_binned_power = best_power_binned[valid_mask]

discrete_peak_idx = np.argmax(valid_binned_power)
window_center = valid_binned_f[discrete_peak_idx]

f_max, f_error, popt = extract_f_max_unbinned_mle(
    f_array=combined_f_array, 
    p_raw=best_power_unbinned, 
    window_center=window_center, 
    window_radius_left=window_radius_left,
    window_radius_right=window_radius_right
)

plot_mle_overlap(
    f_raw=combined_f_array,
    p_raw=best_power_unbinned,
    f_binned=binned_f,
    p_binned=best_power_binned,
    p_err_binned=best_sem_binned,
    popt=popt,
    f_max=f_max,
    f_max_error=f_error,
    window_center=window_center,
    window_radius_left=window_radius_left,
    window_radius_right=window_radius_right
)

plot_max_density(mean_max_density_series, dt, step_skip, mean_cell_density)

linear_smoothed_stack = rep_smoothed_stack[:best_frame_idx]
linear_cell_stack = rep_cell_stack[:best_frame_idx]
linear_binned_spectra = binned_spectra[:best_frame_idx]
linear_binned_sem = binned_sem[:best_frame_idx]

ani = animate_combined(
    f_array=binned_f, 
    smoothed_stack=linear_smoothed_stack, 
    cell_stack=linear_cell_stack, 
    spectra_series=linear_binned_spectra, 
    sem_series=linear_binned_sem, 
    dt=dt, 
    step_skip=step_skip, 
    interval=100
)

if SAVE_VIDEO:
    print("Saving video... This might take a minute.")
    ani.save("simulation_ensemble_evolution.mp4", writer="ffmpeg", fps=30, dpi=200)
    print("Video saved successfully!")
else:
    print("Playing animation interactively...")
    plt.show()