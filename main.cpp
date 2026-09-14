#include <fstream>
#include <iostream>
#include <vector>
#include "tools/SimConfig.h"
#include "tools/wait_closing.h"
#include "tools/math_operation.h"
#include "classes/world_setup.h"
#include "classes/simulation.h"

void export_config_to_json(const SimConfig& cfg, const std::string& filename) {
	std::ofstream out(filename);
	out << "{\n";
	out << "  \"dt\": " << cfg.dt << ",\n";
	out << "  \"total_timesteps\": " << cfg.total_timesteps << ",\n";
	out << "  \"dx\": " << cfg.dx << ",\n";
	out << "  \"width\": " << cfg.width << ",\n";
	out << "  \"height\": " << cfg.height << ",\n";
	out << "  \"diffusion_rate\": " << cfg.diffusion_rate << ",\n";
	out << "  \"decay_rate\": " << cfg.decay_rate << ",\n";
	out << "  \"chem_secretion_rate\": " << cfg.chem_secretion_rate << ",\n";
	out << "  \"total_cells\": " << cfg.total_cells << ",\n";
	out << "  \"chi\": " << cfg.chi << ",\n";
	out << "  \"Dr\": " << cfg.Dr << "\n"; // Notice no comma on the last item
	out << "}\n";
	out.close();
}

int main()
{
	constexpr SimConfig my_config
	{ 
		0.001,    // dt
		2000,   // total_timesteps
		1.0,    // dx
		100,    // width
		100,    // height
		1.0,	// diffusion_rate
		0.1,    // decay_rate
		1.0,	// chem_secretion_rate

		//Cell properties
		100000000,   // total_cells
		2.0,    // chi (Chemotaxis strength)
		0.5     // Dr (Random diffusion) 
	};
	
	export_config_to_json(my_config, "sim_config.json");

	std::random_device rd;
	std::mt19937 rng{ rd() };


	class world_setup world(my_config, rng);
	std::vector<double_vector2d> cells_coordinates{ world.random_cell_distribution() };
	std::vector<double> chem_conc_field{ world.chem_conc_field_initialisation_empty() };

	class simulation sim(my_config, rng, cells_coordinates, chem_conc_field);
	sim.run_simulation();

	wait_closing();

}	
