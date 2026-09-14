import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import imageio_ffmpeg
import json
import scipy.ndimage as ndi

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
    """
    Applies spatial convolution across every time step to create
    a continuous, floating-point density field.
    """
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
    """
    Extracts the peak scalar density (rho_max) at each time step.
    Returns a 1D array of shape (time_steps,).
    """
    # Maximize across the spatial grid (axes 1 and 2: height and width)
    return np.max(smoothed_cell_stack, axis=(1, 2))
        

def compute_radial_power_spectrum(grid):
    grid_fluctuation = grid - np.mean(grid)
    fft2d = np.fft.fft2(grid_fluctuation)
    fft2d_shifted = np.fft.fftshift(fft2d)
    power_spectrum = np.abs(fft2d_shifted)**2
    
    y, x = np.indices(grid.shape)
    center = (grid.shape[0] // 2, grid.shape[1] // 2)
    
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int) 
    
    tbin = np.bincount(r.ravel(), power_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / nr
    
    return radial_profile

def get_power_spectra_series(cell_stack):
    spectra_history = []
    for grid in cell_stack:
        spectrum = compute_radial_power_spectrum(grid)
        spectra_history.append(spectrum)
    return np.array(spectra_history)


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


def animate_combined(smoothed_stack, cell_stack, spectra_series, dt, step_skip, interval):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # --- AX 1: Smoothed Cell Density ---
    smooth_vmin, smooth_vmax = smoothed_stack.min(), smoothed_stack.max()
    img_smooth = ax1.imshow(
        smoothed_stack[0],
        cmap="inferno",
        vmin=smooth_vmin,
        vmax=smooth_vmax,
        origin="lower",
    )
    ax1.set_title("Smoothed Cell Density", fontsize=13)
    fig.colorbar(
        img_smooth, ax=ax1, fraction=0.046, pad=0.04, label="Density (smoothed)"
    )
    ax1.set_xlabel("x coordinate")
    ax1.set_ylabel("y coordinate")

    # --- AX 2: Raw Cell Distribution ---
    cell_vmin, cell_vmax = cell_stack.min(), cell_stack.max()
    img_cell = ax2.imshow(
        cell_stack[0],
        cmap="inferno",
        vmin=cell_vmin,
        vmax=cell_vmax,
        origin="lower",
    )
    ax2.set_title("Raw Cell Distribution", fontsize=13)
    fig.colorbar(
        img_cell, ax=ax2, fraction=0.046, pad=0.04, label="Cell Count"
    )
    ax2.set_xlabel("x coordinate")
    ax2.set_ylabel("y coordinate")

    # --- AX 3: Scalar Power Spectrum ---
    num_bins = spectra_series.shape[1]
    global_max = np.max(spectra_series[:, 1:])

    ax3.set_xlim(1, num_bins // 2)
    ax3.set_ylim(0, global_max * 1.1)
    ax3.set_title("Evolution of Power Spectrum", fontsize=13)
    ax3.set_xlabel("Scalar Wavenumber (k bin)")
    ax3.set_ylabel("Averaged Power")
    ax3.grid(True, linestyle="--", alpha=0.6)

    (line,) = ax3.plot([], [], lw=2.5, color="dodgerblue")
    time_text = ax3.text(
        0.55,
        0.90,
        "",
        transform=ax3.transAxes,
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9
        ),
    )

    plt.tight_layout()

    def init():
        img_smooth.set_data(smoothed_stack[0])
        img_cell.set_data(cell_stack[0])
        line.set_data([], [])
        time_text.set_text("")
        return img_smooth, img_cell, line, time_text

    def update(frame):
        img_smooth.set_data(smoothed_stack[frame])
        img_cell.set_data(cell_stack[frame])
        line.set_data(np.arange(num_bins), spectra_series[frame])
        # Calculate actual time using the time step interval (dt)
        time_text.set_text(f"Time: {frame * step_skip * dt:.2f}")
        return img_smooth, img_cell, line, time_text

    ani = FuncAnimation(
        fig,
        update,
        frames=len(cell_stack),
        init_func=init,
        blit=True,
        interval=interval,
    )

    return ani


# ==========================================
# GLOBAL EXECUTION 
# ==========================================

config, cell_raw = load_data()

# Now you can use them directly for your plots or arrays
width = config["width"]
height = config["height"]
total_cells = config["total_cells"]
total_timesteps = config["total_timesteps"]
dt= config["dt"]
mean_cell_density = total_cells / (width * height)

step_skip = 10

# --- THE SWITCH ---
# Set to True to output an MP4 file. Set to False to play interactively in Spyder.
SAVE_VIDEO = False

cell_stack = cell_raw.reshape(-1, height, width)[::step_skip]
smoothed_stack = get_smoothed_cell_stack(cell_stack, method='box', size=3)
spectra_series = get_power_spectra_series(cell_stack)
max_density_series = get_max_density_series(smoothed_stack)


plot_max_density(max_density_series, dt, step_skip, mean_cell_density)
ani = animate_combined(smoothed_stack, cell_stack, spectra_series, dt, step_skip, interval=50)

# Control flow based on the switch
if SAVE_VIDEO:
    print("Saving video... This might take a minute.")
    ani.save("simulation_evolution.mp4", writer="ffmpeg", fps=30, dpi=200)
    print("Video saved successfully as simulation_evolution.mp4!")
else:
    print("Playing animation interactively...")
    plt.show()