import json
import os
import subprocess
import shutil
import numpy as np

# Anchor everything to the folder containing sweep.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

base_dir = os.path.join(SCRIPT_DIR, "out", "build", "x64-Debug")
results_dir = os.path.join(SCRIPT_DIR, "../", "results")
exe_name = "slime_mold_simulation.exe"
config_name = "sim_config.json"
result_folder_name = "Simulation2_new_Dchem"

num_points = 10

# Define sweep ranges
diffusion_rates = np.linspace(16, 64, num_points)

# Define corresponding linear scaling for total_timesteps
start_timesteps = 300
end_timesteps = 800  # Change this to whatever end value you need
timesteps_array = np.linspace(start_timesteps, end_timesteps, num_points).astype(int)

def run_sweep():
    os.makedirs(results_dir, exist_ok=True)
    
    config_path = os.path.join(base_dir, config_name)
    exe_path = os.path.join(base_dir, exe_name)

    # Verify the executable exists before starting the loop
    if not os.path.isfile(exe_path):
        raise FileNotFoundError(f"Executable not found at: {exe_path}\nMake sure to build the project in Visual Studio first.")

    # 1. Define all simulation parameters here. 
    # Python will use this to rewrite the config file from scratch every time.
    base_config = {
        "dt": 0.00001, 
        "total_timesteps": 2000,  # This gets overwritten in the loop
        "dx": 1.0, 
        "width": 32, 
        "height": 32, 
        "diffusion_rate": 16, 
        "decay_rate": 0.1, 
        "chem_secretion_rate": 1.0, 
        "total_cells": 10240000, 
        "chi": 2.0, 
        "Dr": 0.5,
        "M": 300
    }

    # Zip the two arrays together to iterate through them simultaneously
    for i, (d_rate, t_steps) in enumerate(zip(diffusion_rates, timesteps_array)):
        d_rate_rounded = round(float(d_rate), 2)
        # Ensure it's a standard Python int for JSON serialization
        current_timesteps = int(t_steps) 
        
        print(f"[{i+1}/{num_points}] Running sim: D_chem = {d_rate_rounded}, timesteps = {current_timesteps}")
        
        # 2. Update the diffusion rate AND total_timesteps, then overwrite the JSON
        base_config["diffusion_rate"] = d_rate_rounded
        base_config["total_timesteps"] = current_timesteps
        
        with open(config_path, 'w') as f:
            json.dump(base_config, f, indent=4)

        # 3. Run the C++ Executable
        try:
            # capture_output=False allows C++ cout statements to print to your Python console
            subprocess.run([exe_path], cwd=base_dir, check=True, capture_output=False)
        except subprocess.CalledProcessError as e:
            print(f"Error running executable for D={d_rate_rounded}: {e}")
            continue
            
        # 4. Create unique results folder
        sim_result_folder = os.path.join(results_dir, f"{result_folder_name}_{d_rate_rounded}")
        os.makedirs(sim_result_folder, exist_ok=True)
        
        # 5. Copy files
        files_to_copy = [
            config_name,
            "cell_history.bin",
            "chem_conc_history.bin"
        ]
        
        for file in files_to_copy:
            src = os.path.join(base_dir, file)
            dst = os.path.join(sim_result_folder, file)
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"   -> Warning: {file} not found in {base_dir}.")
                
    print("\nParameter sweep completed successfully.")

if __name__ == "__main__":
    run_sweep()