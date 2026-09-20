import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import imageio_ffmpeg
import json
import scipy.ndimage as ndi
from scipy.optimize import minimize


#run this first for animation %matplotlib qt

# Map the ffmpeg executable for saving videos
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

def load_data(folder_name):
    script_dir = Path(__file__).resolve().parent
    warehouse_path = (script_dir / ".." / "results" / folder_name).resolve()
    
    config_path = warehouse_path / "sim_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
        
    cell_raw = np.fromfile(warehouse_path / "cell_history.bin", dtype=np.int32)
    
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

def extract_f_max_unbinned_mle(f_array, p_raw, window_center, window_radius, plot_fit=True):
    """
    Fits the theoretical Gaussian directly to the raw, unbinned Fourier pixels.
    """
    # 1. Isolate the raw pixels falling within our fitting window
    # We ignore the zero-frequency (DC) component by enforcing f >= 1e-5
    mask = (f_array >= max(1e-5, window_center - window_radius)) & (f_array <= (window_center + window_radius))
    
    f_fit = f_array[mask]
    p_fit = p_raw[mask]
    
    if len(f_fit) < 10:
        print("Warning: Not enough unbinned pixels in window to run MLE.")
        return window_center, 0.0

    # 2. The True Unbinned Negative Log-Likelihood Function (Exponential Distribution)
    def nll_unbinned(params):
        amp, mean, stddev, offset = params
        
        # Enforce physical constraints
        if amp <= 0 or stddev <= 0 or offset <= 0:
            return np.inf
            
        # S evaluates the theoretical Gaussian curve at the EXACT float distance of every pixel
        S = gaussian_model(params, f_fit)
        
        # Exact Exponential Log-Likelihood (no N_i or binning needed)
        return np.sum(np.log(S) + (p_fit / S))

    # 3. Dynamic initial guesses based on the raw pixels in the window
    amp_guess = np.max(p_fit) - np.min(p_fit)
    mean_guess = window_center
    stddev_guess = window_radius / 2.0
    offset_guess = np.maximum(np.min(p_fit), 1e-5)
    
    initial_guess = [amp_guess, mean_guess, stddev_guess, offset_guess]
    
    # 4. Optimizer Bounds
    bounds = [
        (1e-5, np.inf),             # Amplitude > 0
        (f_fit.min(), f_fit.max()), # Mean bounded strictly within fit window
        (1e-5, np.inf),             # StdDev > 0
        (1e-5, np.inf)              # Offset > 0
    ]
    
    # Run the MLE Optimizer
    result = minimize(nll_unbinned, initial_guess, bounds=bounds, method='L-BFGS-B')
    
    if result.success:
        f_max_continuous = result.x[1]
        cov_matrix = result.hess_inv(np.eye(4)) # Extract inverse Hessian for rigorous errors
        f_max_error = np.sqrt(cov_matrix[1, 1])
        print("-> Unbinned MLE Optimization Succeeded.")
    else:
        print("\n[!] Unbinned MLE Failed to converge. Resorting to discrete guess.")
        result.x = initial_guess
        f_max_continuous = mean_guess
        f_max_error = 0.0

    print("\n" + "="*50)
    print(" TRUE UNBINNED MLE f_max EXTRACTION")
    print("="*50)
    print(f"Window Center Guess : f = {window_center:.4f}")
    print(f"Continuous Peak     : f_max = {f_max_continuous:.4f} ± {f_max_error:.4f}")
    print("="*50 + "\n")

    if plot_fit:
        plt.figure(figsize=(8, 5))
        
        # Plot the raw, unbinned pixels!
        plt.scatter(f_fit, p_fit, color='dodgerblue', s=5, alpha=0.3, label='Raw Unbinned Pixels')
                     
        x_smooth = np.linspace(f_fit.min(), f_fit.max(), 200)
        y_smooth = gaussian_model(result.x, x_smooth)
        
        plt.plot(x_smooth, y_smooth, color='crimson', lw=2.5, 
                 label=f'MLE Gaussian Fit\n$f_{{max}} = {f_max_continuous:.4f} \pm {f_max_error:.4f}$')
        
        plt.axvline(f_max_continuous, color='crimson', linestyle='--', alpha=1.0)
        
        plt.title('True Unbinned Maximum Likelihood Fit')
        plt.xlabel('Spatial Frequency $f$ ($1/\lambda$)')
        plt.ylabel('Power (Raw pixels)')
        
        plt.ylim(0, np.max(p_fit) * 1.1)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
    
    return f_max_continuous, f_max_error

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
# ==========================================
# GLOBAL EXECUTION 
# ==========================================
FOLDER_NAME = "sim6"
config, cell_raw = load_data(FOLDER_NAME)

width = config["width"]
height = config["height"]
total_cells = config["total_cells"]
total_timesteps = config["total_timesteps"]
dt = config["dt"]
mean_cell_density = total_cells / (width * height)

step_skip = 10
SAVE_VIDEO = False

cell_stack = cell_raw.reshape(-1, height, width)[::step_skip]
smoothed_stack = get_smoothed_cell_stack(cell_stack, method='box', size=3)
max_density_series = get_max_density_series(smoothed_stack)
# Threading the k_array out of the series function
f_array, spectra_series = get_power_spectra_series(cell_stack)

binned_f, binned_spectra, binned_sem = bin_spectra_series(
    f_array, 
    spectra_series, 
    bin_width=(1.0 / width) 
)

# 3. Animate using the binned arrays
ani = animate_combined(
    f_array=binned_f, 
    smoothed_stack=smoothed_stack, 
    cell_stack=cell_stack, 
    spectra_series=binned_spectra, 
    sem_series=binned_sem, 
    dt=dt, 
    step_skip=step_skip, 
    interval=100
)

if SAVE_VIDEO:
    print("Saving video... This might take a minute.")
    ani.save("simulation_evolution.mp4", writer="ffmpeg", fps=30, dpi=200)
    print("Video saved successfully as simulation_evolution.mp4!")
else:
    print("Playing animation interactively...")
    plt.show()


# --- FIND THE BEST FRAME ---
threshold = mean_cell_density * 1.1
valid_indices = np.where(max_density_series < threshold)[0]

if len(valid_indices) == 0:
    print("Warning: Entire simulation exceeded the 10% non-linear threshold.")
    best_frame_idx = 0
else:
    best_frame_idx = valid_indices[-1]
    
print(f"\nBest Linear Frame Identified: Index {best_frame_idx} (Time: {best_frame_idx * step_skip * dt:.3f})")

# Extract the unbinned data for the best frame
best_power_unbinned = spectra_series[best_frame_idx]

# 1. Use the binned array to find a smart initial guess for the peak center
bin_width = 1.0 / width
nyquist_idx = len(binned_f) // 2
valid_binned_power = binned_spectra[best_frame_idx][1:nyquist_idx]

# Find the center of the highest bin
discrete_peak_idx = np.argmax(valid_binned_power) + 1 
window_center = binned_f[discrete_peak_idx]

# 2. Run the TRUE UNBINNED MLE on the raw pixels in a window around that center
f_max, f_error = extract_f_max_unbinned_mle(
    f_array=f_array, 
    p_raw=best_power_unbinned, 
    window_center=window_center, 
    window_radius=(bin_width * 4) # Look 4 bins in both directions
)

# Visualisation 
plot_max_density(max_density_series, dt, step_skip, mean_cell_density)

