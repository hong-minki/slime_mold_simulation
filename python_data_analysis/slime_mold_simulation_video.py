import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import imageio_ffmpeg
import json
import scipy.ndimage as ndi

# Uncomment this if running in Jupyter/IPython for popup windows
# %matplotlib qt

# Map the ffmpeg executable for saving videos
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

def load_data(folder_path):
    config_path = folder_path / "sim_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
        
    cell_raw = np.fromfile(folder_path / "cell_history.bin", dtype=np.int32)
    
    # NEW: Load chemical field data
    # (Change dtype to np.float32 if your simulation saves it as single-precision)
    chem_file = folder_path / "chem_conc_history.bin"
    if chem_file.exists():
        chem_raw = np.fromfile(chem_file, dtype=np.float64)
    else:
        print(f"Warning: {chem_file.name} not found. Using an array of zeros.")
        chem_raw = np.zeros_like(cell_raw, dtype=np.float64)
        
    return config, cell_raw, chem_raw

def get_smoothed_cell_stack(cell_stack, method='box', size=3):
    smoothed_stack = []
    for grid in cell_stack:
        grid_float = grid.astype(np.float64)
        if method == 'box':
            smoothed_grid = ndi.uniform_filter(grid_float, size=size, mode='wrap')
        elif method == 'gaussian':
            smoothed_grid = ndi.gaussian_filter(grid_float, sigma=size, mode='wrap')
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
        if f_array is None:
            f_array = f_arr
        spectra_history.append(spectrum)
    return f_array, np.array(spectra_history)

def bin_spectra_series(f_array, spectra_series, bin_width=None):
    min_points = 3
    epsilon = 1e-10
    if bin_width is None:
        bin_width = np.min(f_array[f_array > 0])
    
    f_binned = np.floor(f_array / bin_width).astype(int)
    max_bin = np.max(f_binned)
    nr = np.bincount(f_binned, minlength=max_bin + 1)
    
    valid_bins = nr > 0
    enough_points = nr >= min_points
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
        
        sem[enough_points] = np.sqrt(variance[enough_points]) / np.sqrt(nr[enough_points]) + epsilon
        sem[~enough_points] = np.inf 
        
        binned_spectra_series.append(radial_profile)
        binned_sem_series.append(sem)
        
    return binned_f, np.array(binned_spectra_series), np.array(binned_sem_series)

def plot_max_density(max_density_series, dt, step_skip, rho_0=None):
    time_steps = np.arange(len(max_density_series)) * dt
    plt.figure(figsize=(8, 5))
    plt.plot(time_steps * step_skip, max_density_series, color='royalblue', lw=2, label=r'$\max(\rho)$')
    
    if rho_0 is not None:
        plt.axhline(1.1 * rho_0, color='crimson', linestyle='--', label=r'10% Threshold ($1.1\rho_0$)')
        plt.axhline(rho_0, color='gray', linestyle=':', label=r'Mean Density ($\rho_0$)')
        
    plt.xlabel('Time')
    plt.ylabel('Maximum Local Density')
    plt.title('Evolution of Peak Density')
    plt.grid(True, linestyle='--', alpha=0.6)
    if rho_0 is not None:
        plt.legend()
    plt.tight_layout()
    plt.show()

def animate_combined(f_array, chem_stack, smoothed_stack, cell_stack, spectra_series, sem_series, dt, step_skip, interval):
    # Changed to a 2x2 grid for better screen real estate
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    ax_chem, ax_smooth, ax_cell, ax_spec = axs.flatten()

    # 1. Chemical Field
    chem_vmin, chem_vmax = chem_stack.min(), chem_stack.max()
    img_chem = ax_chem.imshow(chem_stack[0], cmap="viridis", vmin=chem_vmin, vmax=chem_vmax, origin="lower")
    ax_chem.set_title("Chemical Concentration", fontsize=13)
    fig.colorbar(img_chem, ax=ax_chem, fraction=0.046, pad=0.04, label="Concentration")
    ax_chem.set_xticks([])
    ax_chem.set_yticks([])

    # 2. Smoothed Cell Density
    smooth_vmin, smooth_vmax = smoothed_stack.min(), smoothed_stack.max()
    img_smooth = ax_smooth.imshow(smoothed_stack[0], cmap="inferno", vmin=smooth_vmin, vmax=smooth_vmax, origin="lower")
    ax_smooth.set_title("Smoothed Cell Density", fontsize=13)
    fig.colorbar(img_smooth, ax=ax_smooth, fraction=0.046, pad=0.04, label="Density (smoothed)")
    ax_smooth.set_xticks([])
    ax_smooth.set_yticks([])

    # 3. Raw Cell Distribution
    cell_vmin, cell_vmax = cell_stack.min(), cell_stack.max()
    img_cell = ax_cell.imshow(cell_stack[0], cmap="inferno", vmin=cell_vmin, vmax=cell_vmax, origin="lower")
    ax_cell.set_title("Raw Cell Distribution", fontsize=13)
    fig.colorbar(img_cell, ax=ax_cell, fraction=0.046, pad=0.04, label="Cell Count")
    ax_cell.set_xticks([])
    ax_cell.set_yticks([])

    # 4. Power Spectrum
    num_bins = spectra_series.shape[1]
    global_max = np.max(spectra_series[:, 1:])
    
    ax_spec.set_xlim(f_array[1], f_array[-1])
    ax_spec.set_ylim(0, global_max * 1.3)
    ax_spec.set_title("Evolution of Power Spectrum", fontsize=13)
    ax_spec.set_xlabel("Spatial Frequency $f$ ($1/\lambda$)")
    ax_spec.set_ylabel("Averaged Power")
    ax_spec.grid(True, linestyle="--", alpha=0.6)

    line, caplines, barlinecols = ax_spec.errorbar(
        f_array, np.zeros(num_bins), yerr=np.zeros(num_bins), 
        fmt='-', lw=2.5, color="dodgerblue", elinewidth=1.5, alpha=0.8
    )
    error_lines = barlinecols[0] 

    time_text = ax_spec.text(
        0.55, 0.90, "", transform=ax_spec.transAxes, fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
    )

    plt.tight_layout()

    def init():
        img_chem.set_data(chem_stack[0])
        img_smooth.set_data(smoothed_stack[0])
        img_cell.set_data(cell_stack[0])
        line.set_data([], [])
        empty_segments = [np.array([[x, 0], [x, 0]]) for x in f_array]
        error_lines.set_segments(empty_segments)
        time_text.set_text("")
        return img_chem, img_smooth, img_cell, line, error_lines, time_text

    def update(frame):
        img_chem.set_data(chem_stack[frame])
        img_smooth.set_data(smoothed_stack[frame])
        img_cell.set_data(cell_stack[frame])
        
        y = spectra_series[frame]
        err = sem_series[frame].copy()
        err[np.isinf(err)] = np.nan
        
        line.set_data(f_array, y)
        segments = [np.array([[x, y_val - e], [x, y_val + e]]) for x, y_val, e in zip(f_array, y, err)]
        error_lines.set_segments(segments)
        
        time_text.set_text(f"Time: {frame * step_skip * dt:.2f}")
        return img_chem, img_smooth, img_cell, line, error_lines, time_text

    ani = FuncAnimation(fig, update, frames=len(cell_stack), init_func=init, blit=True, interval=interval)
    return ani

# ==========================================
# VISUALIZATION EXECUTION 
# ==========================================
if __name__ == "__main__":
    
    # --- CHOOSE YOUR D_CHEM TO ANIMATE ---
    D_chem_list = np.linspace(16, 64, 20)
    # Pick a specific index here (e.g., 1 for 18.53, 19 for 64.0)
    D_chem = round(D_chem_list[0], 2) 
    
    SAVE_VIDEO = False
    LINEAR_PLOT = False  
    step_skip = 10

    PATTERN = f"Simulation*_new_Dchem_{D_chem}" 
    script_dir = Path(__file__).resolve().parent
    results_dir = (script_dir / ".." / ".." / "results").resolve()
    folder_paths = sorted(list(results_dir.glob(PATTERN)))

    if not folder_paths:
        raise ValueError(f"No folders found matching {PATTERN} in {results_dir}")

    print(f"Loading data for D_chem = {D_chem} ({len(folder_paths)} simulation(s))...")

    all_max_densities = []
    all_spectra_series = []
    rep_chem_stack = None
    rep_cell_stack = None
    rep_smoothed_stack = None
    f_array_single = None
    config = None

    for i, folder_path in enumerate(folder_paths):
        c, cell_raw, chem_raw = load_data(folder_path)
        
        if config is None:
            config = c
            width = config["width"]
            height = config["height"]
            total_cells = config["total_cells"]
            dt = config["dt"]
            mean_cell_density = total_cells / (width * height)
            
        c_stack = cell_raw.reshape(-1, height, width)[::step_skip]
        chem_stack = chem_raw.reshape(-1, height, width)[::step_skip]
        
        all_max_densities.append(get_max_density_series(c_stack))
        
        f_arr, spec_series = get_power_spectra_series(c_stack)
        all_spectra_series.append(spec_series)
        
        if f_array_single is None:
            f_array_single = f_arr
        if i == 0:
            rep_chem_stack = chem_stack
            rep_cell_stack = c_stack
            rep_smoothed_stack = get_smoothed_cell_stack(c_stack, method='box', size=3)

    min_frames = min([len(dens) for dens in all_max_densities])
    all_max_densities = [dens[:min_frames] for dens in all_max_densities]
    all_spectra_series = [spec[:min_frames, :] for spec in all_spectra_series]
    rep_chem_stack = rep_chem_stack[:min_frames]
    rep_cell_stack = rep_cell_stack[:min_frames]
    rep_smoothed_stack = rep_smoothed_stack[:min_frames]

    mean_max_density_series = np.mean(all_max_densities, axis=0)
    combined_f_array = np.tile(f_array_single, len(folder_paths))
    combined_spectra_series = np.concatenate(all_spectra_series, axis=1)

    bin_width = 1 / width
    binned_f, binned_spectra, binned_sem = bin_spectra_series(combined_f_array, combined_spectra_series, bin_width=bin_width)

    # Plot Density
    plot_max_density(mean_max_density_series, dt, step_skip, mean_cell_density)

    # Determine frames for animation
    if LINEAR_PLOT:
        threshold = mean_cell_density * 1.1
        valid_indices = np.where(mean_max_density_series < threshold)[0]
        best_frame_idx = valid_indices[-1] if len(valid_indices) > 0 else 0
        
        ani_chem_stack = rep_chem_stack[:best_frame_idx]
        ani_smoothed_stack = rep_smoothed_stack[:best_frame_idx]
        ani_cell_stack = rep_cell_stack[:best_frame_idx]
        ani_binned_spectra = binned_spectra[:best_frame_idx]
        ani_binned_sem = binned_sem[:best_frame_idx]
    else:
        ani_chem_stack = rep_chem_stack
        ani_smoothed_stack = rep_smoothed_stack
        ani_cell_stack = rep_cell_stack
        ani_binned_spectra = binned_spectra
        ani_binned_sem = binned_sem

    # Create Animation
    print("Generating animation...")
    ani = animate_combined(
        f_array=binned_f, 
        chem_stack=ani_chem_stack,
        smoothed_stack=ani_smoothed_stack, 
        cell_stack=ani_cell_stack, 
        spectra_series=ani_binned_spectra, 
        sem_series=ani_binned_sem, 
        dt=dt, 
        step_skip=step_skip, 
        interval=100
    )
    
    if SAVE_VIDEO:
        print("Saving video... This might take a minute.")
        ani.save(f"simulation_Dchem_{D_chem}_evolution.mp4", writer="ffmpeg", fps=30, dpi=200)
        print("Video saved successfully!")
    else:
        print("Playing animation interactively...")
        plt.show()