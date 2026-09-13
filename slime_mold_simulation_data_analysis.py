import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import imageio_ffmpeg
import json


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

def animate_combined(cell_stack, spectra_series, step_skip, interval):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- AX 1: Real Space ---
    cell_vmin, cell_vmax = cell_stack.min(), cell_stack.max()
    img_cell = ax1.imshow(cell_stack[0], cmap='inferno', vmin=cell_vmin, vmax=cell_vmax, origin='lower')
    ax1.set_title("Cell Distribution", fontsize=14)
    fig.colorbar(img_cell, ax=ax1, fraction=0.046, pad=0.04, label="Cell Density")
    ax1.set_xlabel("x coordinate")
    ax1.set_ylabel("y coordinate")
    
    # --- AX 2: Fourier Space ---
    num_bins = spectra_series.shape[1]
    global_max = np.max(spectra_series[:, 1:]) 
    
    ax2.set_xlim(1, num_bins // 2) 
    ax2.set_ylim(0, global_max * 1.1)
    ax2.set_title("Evolution of Scalar Power Spectrum", fontsize=14)
    ax2.set_xlabel("Scalar Wavenumber (k bin)")
    ax2.set_ylabel("Averaged Power")
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    line, = ax2.plot([], [], lw=2.5, color='dodgerblue')
    time_text = ax2.text(0.65, 0.90, '', transform=ax2.transAxes, 
                         fontsize=12, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
    
    plt.tight_layout()
    
    def init():
        img_cell.set_data(cell_stack[0])
        line.set_data([], [])
        time_text.set_text('')
        return img_cell, line, time_text

    def update(frame):
        img_cell.set_data(cell_stack[frame])
        line.set_data(np.arange(num_bins), spectra_series[frame])
        time_text.set_text(f'Time Step: {frame * step_skip}')
        return img_cell, line, time_text

    ani = FuncAnimation(fig, update, frames=len(cell_stack),
                        init_func=init, blit=True, interval=interval)
    
    # Removed plt.show() from here so the global switch can control playback
    return ani


# ==========================================
# GLOBAL EXECUTION 
# ==========================================

config, cell_raw = load_data()

# Now you can use them directly for your plots or arrays
width = config["width"]
height = config["height"]
total_cells = config["total_cells"]
mean_cell_density = total_cells / (width * height)

step_skip = 10

# --- THE SWITCH ---
# Set to True to output an MP4 file. Set to False to play interactively in Spyder.
SAVE_VIDEO = False

cell_stack = cell_raw.reshape(-1, height, width)[::step_skip]
spectra_series = get_power_spectra_series(cell_stack)

ani = animate_combined(cell_stack, spectra_series, step_skip, interval=50)

# Control flow based on the switch
if SAVE_VIDEO:
    print("Saving video... This might take a minute.")
    ani.save("simulation_evolution.mp4", writer="ffmpeg", fps=30, dpi=200)
    print("Video saved successfully as simulation_evolution.mp4!")
else:
    print("Playing animation interactively...")
    plt.show()