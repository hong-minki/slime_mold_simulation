#include "simulation.h"

simulation::simulation(const SimConfig& incoming_config, std::vector<double_vector2d>& incoming_cells_coordinate, std::vector<double>& incoming_chem_conc_field)
	:my_config(incoming_config), cells_coordinate(incoming_cells_coordinate), chem_conc_field(incoming_chem_conc_field),
	num_cells(incoming_config.total_grid_points(), 0)
{
}

void simulation::cells_update()
{

}


void simulation::diffusion_calculation()
{
	std::vector<double> next_chem_conc_field{ chem_conc_field };

	for (int y = 0; y < my_config.height; ++y)
	{
		for (int x = 0; x < my_config.width; ++x)
		{
			int_vector2d coordinate_vector = { x, y };

			int center_index = get_index(coordinate_vector, my_config.width, my_config.height);
			double chem_conc_new = chem_conc_field[center_index] + my_config.dt * (my_config.diffusion_rate * discrete_laplacian(my_config, coordinate_vector, chem_conc_field)
				- my_config.decay_rate * chem_conc_field[center_index]
				+ my_config.chem_secretion_rate * num_cells[center_index]);
			if (chem_conc_new < 0.0) chem_conc_new = 0.0;
			next_chem_conc_field[center_index] = chem_conc_new;
		}
	}
	chem_conc_field.swap(next_chem_conc_field);
}

void simulation::run_simulation()
{
	std::ofstream outfile("chem_conc_history.bin", std::ios::binary);
	
	//initialisation of fields
	std::vector<int> num_cells(my_config.total_grid_points(), 0);

	for (int timestep = 0; timestep < my_config.total_timesteps; ++timestep)
	{
		diffusion_calculation();

		outfile.write(reinterpret_cast<const char*>(chem_conc_field.data()),
			chem_conc_field.size() * sizeof(double));

		if (timestep % 100 == 0)
		{
			std::cout << "Timestep: " << timestep << std::endl;
		}
	}
	outfile.close();
}