# Slime Mold Chemotaxis Simulation

This ongoing project simulates the biological phenomenon of chemotaxis, where cell agents move in response to chemical gradients. The numerical simulation is calculated in C++ for optimal performance, while the visualisation and data analysis are handled by separate Python scripts.

## Core Physics & Mechanics
*   **Cell Dynamics:** Cell movement is modeled using the Overdamped Langevin equation, balancing directed chemotactic drift ($\chi$) against random diffusion noise ($D_r$). The equation is given as
    
    $\mathbf{r}(t + \Delta t) = \mathbf{r}(t) + \chi \nabla c(\mathbf{r}) \Delta t + \sqrt{2 D_r \Delta t} \eta$

*   **Chemical Diffusion:** The environmental chemical field updates via a discrete Laplacian calculation. The field continuously evolves based on a defined diffusion rate, natural decay rate, and new chemical secretions from the cell agents present on the grid. The equation is given as
    
    $\frac{\partial c(r, t)}{\partial t} = D \nabla^2 c(r, t) - \gamma c(r, t) + \alpha \rho(r, t)$

*   **Data Export:** The simulation writes frame-by-frame grid data directly into raw binary files.

## Project Structure
Based on the current repository layout, the project is divided into the following distinct directories and files:

*   **Root Files:**
    *   `main.cpp` / `main.h`: The primary entry points defining the hyperparameters (e.g., total cells, timesteps).
    *   `sweep.py`: A Python script utilized in the root directory for parameter sweeping.
    *   `slime_mold_simulation_report_v1.pdf`: The finalized report documenting the simulation results.
*   **`classes/`**:
    *   `simulation.cpp` / `.h`: Handles the main execution loop, cell coordinate updates, and diffusion mathematics.
    *   `world_setup.cpp` / `.h`: Initialises the random uniform cell distribution and sets up the starting chemical fields.
*   **`tools/`**:
    *   `math_operation.cpp` / `.h`: Contains inline utility functions for discrete Laplacian and gradient calculations.
    *   `SimConfig.h`: A structural wrapper holding all runtime physical constants.
    *   `wait_closing.cpp` / `.h`: A simple utility to prevent the terminal window from closing immediately after execution.
*   **`python_data_analysis/`**: 
    *   Contains Python scripts for data processing and visualization, including `calculation.py`, `cluster_separation_plots.py`, `slime_mold_simulation_data_analysis.py`, and `slime_mold_simulation_video.py`.
    *   Stores exported data files such as `cluster_separation_data.csv` and `.xlsx` variants.
*   **`DOCUMENTS/`**:
    *   Contains the laboratory notes (`slime_mold_simulation_lab_notes.md`) and various output plots (e.g., `cluster_separation_results.png`, `example_MLE_fitting.png`).

## Getting Started
1. **Compile the Core:** Build the C++ project using the provided `CMakeLists.txt`.
2. **Run the Simulation:** Execute the compiled C++ binary. This will step through the simulation and output the necessary `.bin` files into your build directory.
3. **Render the Video:** Ensure `numpy`, `matplotlib`, and `imageio_ffmpeg` are installed in your Python environment. Navigate to the `python_data_analysis/` directory and run `slime_mold_simulation_video.py` to parse the binary data and save the animation to disk.