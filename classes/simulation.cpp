#include "simulation.h"

simulation::simulation(const SimConfig& incoming_config, std::vector<double_vector2d>& incoming_cells_coordinate, std::vector<double>& incoming_chem_conc_field)
	:my_config(incoming_config), cells_coordinate(incoming_cells_coordinate), chem_conc_field(incoming_chem_conc_field)
{
}


void simulation::run_simulation()
{
	std::ofstream outfile("chem_conc_history.bin", std::ios::binary);
	
	//initialisation of fields
	std::vector<int> num_cells(my_config.total_grid_points(), 0);

	for (int timestep = 0; timestep < my_config.total_timesteps; ++timestep)
	{
		diffusion_calculation(my_config, chem_conc_field, num_cells);

		outfile.write(reinterpret_cast<const char*>(chem_conc_field.data()),
			chem_conc_field.size() * sizeof(double));

		if (timestep % 100 == 0)
		{
			std::cout << "Timestep: " << timestep << std::endl;
		}
	}
	outfile.close();
}