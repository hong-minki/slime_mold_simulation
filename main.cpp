#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <regex>
#include <stdexcept>

#include "tools/SimConfig.h"
#include "tools/wait_closing.h"
#include "tools/math_operation.h"
#include "classes/world_setup.h"
#include "classes/simulation.h"

// Function to read config from JSON without external libraries
SimConfig load_config_from_json(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open " + filename);
    }

    SimConfig cfg{};
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    // Helper lambdas to extract values using regex
    auto extract_double = [&](const std::string& key) {
        // Added (?:[eE][-+]?[0-9]+)? to support scientific notation like 1e-05
        std::regex r("\"" + key + "\"\\s*:\\s*([-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?)");
        std::smatch match;
        if (std::regex_search(content, match, r)) return std::stod(match[1].str());
        return 0.0;
        };

    auto extract_int = [&](const std::string& key) {
        std::regex r("\"" + key + "\"\\s*:\\s*([-+]?[0-9]+)");
        std::smatch match;
        if (std::regex_search(content, match, r)) return std::stoi(match[1].str());
        return 0;
        };

    cfg.dt = extract_double("dt");
    cfg.total_timesteps = extract_int("total_timesteps");
    cfg.dx = extract_double("dx");
    cfg.width = extract_int("width");
    cfg.height = extract_int("height");
    cfg.diffusion_rate = extract_double("diffusion_rate");
    cfg.decay_rate = extract_double("decay_rate");
    cfg.chem_secretion_rate = extract_double("chem_secretion_rate");
    cfg.total_cells = extract_int("total_cells");
    cfg.chi = extract_double("chi");
    cfg.Dr = extract_double("Dr");

    return cfg;
}

int main()
{
    SimConfig my_config;
    try {
        // Reads from the directory the executable is running in
        my_config = load_config_from_json("sim_config.json");
    }
    catch (const std::exception& e) {
        std::cerr << "Error reading config: " << e.what() << "\n";
        return 1;
    }

    std::random_device rd;
    std::mt19937 rng{ rd() };

    class world_setup world(my_config, rng);
    std::vector<double_vector2d> cells_coordinates{ world.random_cell_distribution() };
    std::vector<double> chem_conc_field{ world.chem_conc_field_initialisation_empty() };

    class simulation sim(my_config, rng, cells_coordinates, chem_conc_field);
    sim.run_simulation();

    // WARNING: Removed wait_closing() so the Python script doesn't hang!
    // wait_closing(); 

    return 0;
}