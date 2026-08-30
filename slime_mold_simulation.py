import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

import imageio_ffmpeg
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

# 1. Locate file
script_dir = Path(__file__).resolve().parent
warehouse_path = script_dir / "out" / "build" / "x64-Debug"

width = 100
height = 100
frame_step = 10  # <-- Change this to skip more or fewer frames

# 2. Load, reshape, and SLICE binary data
chem_raw = np.fromfile(warehouse_path / "chem_conc_history.bin", dtype=np.float64)
# [::frame_step] takes 1 out of every N frames
chem_stack = chem_raw.reshape(-1, height, width)[::frame_step]

cell_raw = np.fromfile(warehouse_path / "cell_history.bin", dtype=np.int32)
cell_stack = cell_raw.reshape(-1, height, width)[::frame_step]

total_frames = chem_stack.shape[0]

# 3. Setup figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

chem_vmin, chem_vmax = chem_stack.min(), chem_stack.max()
cell_vmin, cell_vmax = cell_stack.min(), cell_stack.max()

img_chem = ax1.imshow(chem_stack[0], cmap='viridis', vmin=chem_vmin, vmax=chem_vmax, origin='lower')
title_chem = ax1.set_title("Chemicals | Step: 0")
plt.colorbar(img_chem, ax=ax1, fraction=0.046, pad=0.04)

img_cell = ax2.imshow(cell_stack[0], cmap='magma', vmin=cell_vmin, vmax=cell_vmax, origin='lower')
title_cell = ax2.set_title("Cells | Step: 0")
plt.colorbar(img_cell, ax=ax2, fraction=0.046, pad=0.04)

plt.tight_layout()

# 4. Animation update function
def update(frame):
    img_chem.set_data(chem_stack[frame])
    img_cell.set_data(cell_stack[frame])
    
    # Calculate actual simulation timestep for accurate titles
    actual_step = frame * frame_step
    title_chem.set_text(f"Chemicals | Step: {actual_step}")
    title_cell.set_text(f"Cells | Step: {actual_step}")
    
    return img_chem, img_cell, title_chem, title_cell

# 5. Create and run animation
anim = FuncAnimation(fig, update, frames=total_frames, interval=20, blit=True)
anim.save("simulation.mp4", writer="ffmpeg", fps=30, dpi=200)

print("Video saved successfully!")