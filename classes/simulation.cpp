#include "simulation.h"

simulation::simulation(const SimConfig& incoming_config, std::mt19937& incoming_rng, std::vector<double_vector2d>& incoming_cells_coordinates, std::vector<double>& incoming_chem_conc_field)
	:my_config(incoming_config), rng(incoming_rng), cells_coordinates(incoming_cells_coordinates), chem_conc_field(incoming_chem_conc_field),
	num_cells(incoming_config.total_grid_points(), 0)
{
	map_cells_to_grid();
}

void simulation::map_cells_to_grid()
{
	// Wipe the grid clean
	std::fill(num_cells.begin(), num_cells.end(), 0);

	// Bin every cell into its grid square based on its current X/Y
	for (const auto& cell_coordinates : cells_coordinates)
	{
		int grid_x{ static_cast<int>(cell_coordinates.x / my_config.dx) };
		int grid_y{ static_cast<int>(cell_coordinates.y / my_config.dx) };

		int cell_index{ get_index({ grid_x, grid_y }, my_config.width, my_config.height) };

		num_cells[cell_index] += 1;
	}
}

void simulation::cells_update()
{
	std::normal_distribution<double> eta(0.0, 1.0);
	double physical_width{ my_config.width * my_config.dx };
	double physical_height{ my_config.height * my_config.dx };
	double drift_scale{ my_config.chi * my_config.dt };
	double noise_scale{ std::sqrt(2.0 * my_config.Dr * my_config.dt) };

	// 1. Do the physics (move the cells)
	for (auto& cell_coordinates : cells_coordinates)
	{
		double_vector2d chem_field_gradient{ discrete_gradient(my_config, cell_coordinates, chem_conc_field) };

		cell_coordinates.x += (drift_scale * chem_field_gradient.x) + (noise_scale * eta(rng));
		cell_coordinates.y += (drift_scale * chem_field_gradient.y) + (noise_scale * eta(rng));

		// Apply PBC
		cell_coordinates.x = std::fmod(std::fmod(cell_coordinates.x, physical_width) + physical_width, physical_width);
		cell_coordinates.y = std::fmod(std::fmod(cell_coordinates.y, physical_height) + physical_height, physical_height);
	}

	// 2. Re-map the new positions to the grid
	map_cells_to_grid();
}


void simulation::diffusion_calculation()
{
	std::vector<double> next_chem_conc_field{ chem_conc_field };

	for (int y = 0; y < my_config.height; ++y)
	{
		for (int x = 0; x < my_config.width; ++x)
		{
			int_vector2d coordinates_vector = { x, y };

			int center_index = get_index(coordinates_vector, my_config.width, my_config.height);
			double chem_conc_new = chem_conc_field[center_index] + my_config.dt * my_config.M * (my_config.diffusion_rate * discrete_laplacian(my_config, coordinates_vector, chem_conc_field)
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
	// Open two separate files for clean data separation
	std::ofstream chem_outfile("chem_conc_history.bin", std::ios::binary);
	std::ofstream cells_outfile("cell_history.bin", std::ios::binary);

	// Save initial state before the simulation loop
	chem_outfile.write(reinterpret_cast<const char*>(chem_conc_field.data()),
		chem_conc_field.size() * sizeof(double));
	cells_outfile.write(reinterpret_cast<const char*>(num_cells.data()),
		num_cells.size() * sizeof(int));

	for (int timestep = 0; timestep < my_config.total_timesteps -1; ++timestep)
	{
		cells_update();
		diffusion_calculation();

		// Write chemical concentration field (array of doubles)
		chem_outfile.write(reinterpret_cast<const char*>(chem_conc_field.data()),
			chem_conc_field.size() * sizeof(double));

		// Write cell count field (array of ints)
		cells_outfile.write(reinterpret_cast<const char*>(num_cells.data()),
			num_cells.size() * sizeof(int));

		if (timestep % 100 == 0)
		{
			std::cout << "Timestep: " << timestep << std::endl;
		}
	}

	chem_outfile.close();
	cells_outfile.close();
}