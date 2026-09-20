import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import imageio_ffmpeg
import json
import scipy.ndimage as ndi
from scipy import odr
from scipy.optimize import curve_fit


#run this first for animation %matplotlib qt

# Map the ffmpeg executable for saving videos
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

def load_data():
    script_dir = Path(__file__).resolve().parent
    warehouse_path = script_dir / "out" / "build" / "x64-Debug"
    
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
        

def compute_radial_power_spectrum(grid, bin_width):
    min_points = 3
    grid_fluctuation = grid - np.mean(grid)
    fft2d = np.fft.fft2(grid_fluctuation)
    fft2d_shifted = np.fft.fftshift(fft2d)
    power_spectrum = np.abs(fft2d_shifted)**2
    
    y, x = np.indices(grid.shape)
    center = (grid.shape[0] // 2, grid.shape[1] // 2)
    
    # 1. Keep distances as floats
    r_float = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    # 2. Scale by bin_width and convert to int for bincount
    r_binned = np.floor(r_float / bin_width).astype(int)
    
    tbin = np.bincount(r_binned.ravel(), power_spectrum.ravel())
    nr = np.bincount(r_binned.ravel())
    
    valid_bins = nr > 0
    radial_profile = np.zeros_like(tbin, dtype=np.float64)
    radial_profile[valid_bins] = tbin[valid_bins] / nr[valid_bins]
    
    tbin_sq = np.bincount(r_binned.ravel(), power_spectrum.ravel()**2)
    mean_sq_power = np.zeros_like(tbin_sq, dtype=np.float64)
    mean_sq_power[valid_bins] = tbin_sq[valid_bins] / nr[valid_bins]
    
    variance = np.maximum(mean_sq_power - radial_profile**2, 0)
    
    epsilon = 1e-10 
    sem = np.zeros_like(variance)
    
    enough_points = nr >= min_points
    sem[enough_points] = np.sqrt(variance[enough_points]) / np.sqrt(nr[enough_points]) + epsilon
    sem[~enough_points] = np.inf 
    
    # 3. Explicitly construct the k-axis map
    k_array = (np.arange(len(radial_profile)) + 0.5) * bin_width
    
    return k_array, radial_profile, sem

def get_power_spectra_series(cell_stack, bin_width=0.5):
    spectra_history = []
    sem_history = []
    k_array = None
    
    for grid in cell_stack:
        k_arr, spectrum, sem = compute_radial_power_spectrum(grid, bin_width)
        
        # Capture the k_array once (it's identical for all frames)
        if k_array is None:
            k_array = k_arr
            
        spectra_history.append(spectrum)
        sem_history.append(sem)
        
    return k_array, np.array(spectra_history), np.array(sem_history)

#Fitting 
def gaussian_model(B, x):
    """Gaussian peak: B[0]=Amplitude, B[1]=Mean(k_max), B[2]=StdDev, B[3]=Offset"""
    return B[0] * np.exp(-((x - B[1])**2) / (2 * B[2]**2)) + B[3]

def extract_k_max_odr(k_array, power, sem, window=4, plot_fit=True):
    from scipy.optimize import curve_fit
    
    # Determine bin width to calculate correct discretization errors
    bin_width = k_array[1] - k_array[0]
    
    # Nyquist limit physically occurs at half the grid size.
    nyquist_idx = len(k_array) // 2 
    
    valid_power = power[1:nyquist_idx]
    k_vals = k_array[1:nyquist_idx]
    sem_vals = sem[1:nyquist_idx]
    
    discrete_idx = np.argmax(valid_power)
    
    start = max(0, discrete_idx - window)
    end = min(len(valid_power), discrete_idx + window + 1)
    
    k_fit = k_vals[start:end]
    p_fit = valid_power[start:end]
    p_err = sem_vals[start:end]
    
    finite_mask = ~np.isinf(p_err)
    k_fit = k_fit[finite_mask]
    p_fit = p_fit[finite_mask]
    p_err = p_err[finite_mask]
    
    if len(k_fit) < 4:
        print("Warning: Not enough finite data points in the window to fit a Gaussian.")
        return float(k_vals[discrete_idx]), bin_width / 2

    # --- Y-AXIS SCALING ---
    y_scale = np.max(p_fit)
    p_fit_scaled = p_fit / y_scale
    
    p_err_scaled = p_err / y_scale
    p_err_scaled = np.maximum(p_err_scaled, 0.01) 

    k_err = np.full_like(k_fit, bin_width / 2, dtype=np.float64)
    
    model = odr.Model(gaussian_model)
    data = odr.RealData(k_fit, p_fit_scaled, sx=k_err, sy=p_err_scaled)
    
    # Dynamic initial guesses
    amp_guess = 1.0 - np.min(p_fit_scaled)
    mean_guess = k_vals[discrete_idx]
    stddev_guess = bin_width * 1.5 
    offset_guess = np.min(p_fit_scaled)
    
    B0 = [amp_guess, mean_guess, stddev_guess, offset_guess]
    
    myodr = odr.ODR(data, model, beta0=B0)
    output = myodr.run()
    
    k_max_continuous = output.beta[1]
    k_max_error = output.sd_beta[1]
    
    # --- FIX 2: Automatic Fallback for Sharp Peaks ---
    if k_max_error == 0.0 or 'Problem is not full rank' in output.stopreason[0]:
        print("\n[!] ODR Failed (Peak too sharp/irregular). Falling back to standard curve_fit...")
        
        # Wrapper to translate between curve_fit format and ODR format
        def curve_fit_gaussian(x, amp, mean, stddev, offset):
            return gaussian_model([amp, mean, stddev, offset], x)
            
        try:
            popt, pcov = curve_fit(
                curve_fit_gaussian, k_fit, p_fit_scaled, p0=B0, 
                sigma=p_err_scaled, absolute_sigma=True, maxfev=5000
            )
            # Create a mock output.beta array so the plotting code still works
            output.beta = popt
            k_max_continuous = popt[1]
            k_max_error = np.sqrt(pcov[1, 1])
            print("-> curve_fit succeeded.")
        except RuntimeError:
            print("-> curve_fit also failed. Resorting to discrete bin maximum.")
            output.beta = B0
            k_max_continuous = mean_guess
            k_max_error = bin_width / 2

    # Re-scale the amplitude and offset back to original magnitude for plotting
    output.beta[0] *= y_scale
    output.beta[3] *= y_scale
    
    print("\n" + "="*50)
    print(" RIGOROUS k_max EXTRACTION")
    print("="*50)
    print(f"Discrete Peak      : k = {k_vals[discrete_idx]}")
    print(f"Continuous Peak    : k_max = {k_max_continuous:.3f} ± {k_max_error:.3f}")
    
    if k_max_continuous <= 0 or k_max_continuous >= k_array[nyquist_idx]:
        print("Warning: Fit failed to converge on a physical peak inside the domain.")
        
    print("="*50 + "\n")

    if plot_fit:
        plt.figure(figsize=(8, 5))
        
        sem_plot = np.where(np.isinf(sem_vals), np.nan, sem_vals)
        
        plt.errorbar(k_vals, valid_power, yerr=sem_plot, fmt='o', 
                     color='lightgray', ecolor='lightgray', elinewidth=1, capsize=2, 
                     label='Data outside fit window')
        
        plt.errorbar(k_fit, p_fit, yerr=p_err, xerr=k_err, fmt='o', 
                     color='dodgerblue', ecolor='dodgerblue', elinewidth=1.5, capsize=3, 
                     label='Data used in fit')
                     
        x_smooth = np.linspace(k_fit.min() - (bin_width * 3), k_fit.max() + (bin_width * 3), 200)
        y_smooth = gaussian_model(output.beta, x_smooth)
        
        plt.plot(x_smooth, y_smooth, color='crimson', lw=2.5, 
                 label=f'Gaussian Fit\n$k_{{max}} = {k_max_continuous:.2f} \pm {k_max_error:.2f}$')
        
        plt.axvline(k_max_continuous, color='crimson', linestyle='--', alpha=0.6)
        
        plt.title('Gaussian Fit of Power Spectrum Peak (Best Frame)')
        plt.xlabel('Scalar Wavenumber (k)')
        plt.ylabel('Averaged Power')
        
        window_width = window * bin_width
        plt.xlim(max(k_array[1], k_vals[discrete_idx] - window_width - 3), 
                 min(k_array[nyquist_idx], k_vals[discrete_idx] + window_width + 3))
                 
        plt.ylim(0, np.max(p_fit) * 1.3)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
    
    return k_max_continuous, k_max_error#Visualisation


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


def animate_combined(k_array, smoothed_stack, cell_stack, spectra_series, sem_series, dt, step_skip, interval):
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
    
    nyquist_idx = len(k_array) // 2

    # Scale X axis limits using the actual k_array bounds
    ax3.set_xlim(k_array[1], k_array[nyquist_idx])
    ax3.set_ylim(0, global_max * 1.3)
    ax3.set_title("Evolution of Power Spectrum", fontsize=13)
    ax3.set_xlabel("Scalar Wavenumber (k)")
    ax3.set_ylabel("Averaged Power")
    ax3.grid(True, linestyle="--", alpha=0.6)

    line, caplines, barlinecols = ax3.errorbar(
        k_array, np.zeros(num_bins), yerr=np.zeros(num_bins), 
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
        
        empty_segments = [np.array([[x, 0], [x, 0]]) for x in k_array]
        error_lines.set_segments(empty_segments)
        
        time_text.set_text("")
        return img_smooth, img_cell, line, error_lines, time_text

    def update(frame):
        img_smooth.set_data(smoothed_stack[frame])
        img_cell.set_data(cell_stack[frame])
        
        y = spectra_series[frame]
        err = sem_series[frame].copy()
        
        err[np.isinf(err)] = np.nan
        
        # Plot against k_array
        line.set_data(k_array, y)
        
        segments = [np.array([[x, y_val - e], [x, y_val + e]]) for x, y_val, e in zip(k_array, y, err)]
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

config, cell_raw = load_data()

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

# Threading the k_array out of the series function
k_array, spectra_series, sem_series = get_power_spectra_series(cell_stack, bin_width=1)

max_density_series = get_max_density_series(smoothed_stack)

# --- FIND THE BEST FRAME ---
threshold = mean_cell_density * 1.1
valid_indices = np.where(max_density_series < threshold)[0]

if len(valid_indices) == 0:
    print("Warning: Entire simulation exceeded the 10% non-linear threshold.")
    best_frame_idx = 0
else:
    best_frame_idx = valid_indices[-1]
    
print(f"\nBest Linear Frame Identified: Index {best_frame_idx} (Time: {best_frame_idx * step_skip * dt:.3f})")

# --- EXTRACT k_max WITH ERROR ---
best_power = spectra_series[best_frame_idx]
best_sem = sem_series[best_frame_idx]

# Pass the k_array into the extraction logic
k_max, k_error = extract_k_max_odr(k_array, best_power, best_sem, window=6)

# Visualisation 
plot_max_density(max_density_series, dt, step_skip, mean_cell_density)

# Pass the k_array into the animation function
ani = animate_combined(k_array, smoothed_stack, cell_stack, spectra_series, sem_series, dt, step_skip, interval=50)

if SAVE_VIDEO:
    print("Saving video... This might take a minute.")
    ani.save("simulation_evolution.mp4", writer="ffmpeg", fps=30, dpi=200)
    print("Video saved successfully as simulation_evolution.mp4!")
else:
    print("Playing animation interactively...")
    plt.show()