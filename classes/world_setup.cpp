#include "world_setup.h"

world_setup::world_setup(const SimConfig& incoming_config, std::mt19937& incoming_rng)
	: my_config(incoming_config),                                    
      rng(incoming_rng),
      dist_x{ 0.0, my_config.width * my_config.dx },
      dist_y{ 0.0, my_config.height * my_config.dx }
{
}


std::vector<double_vector2d> world_setup::random_cell_distribution()
{
    std::vector<double_vector2d> cells_coordinates(my_config.total_cells);
    for (auto& cell_coordinates : cells_coordinates) 
    {
        cell_coordinates.x = dist_x(rng);
        cell_coordinates.y = dist_y(rng);
    }

	return cells_coordinates;
}   


std::vector<double> world_setup::chem_conc_field_initialisation_square()
{
	std::vector<double> chem_conc_field(my_config.total_grid_points(), 0.0);
	
	int center_x = my_config.width / 2;
	int center_y = my_config.height / 2;

	// Loop from -5 to 4 to create exactly a 10x10 grid (10 cells in each direction)
	for (int dy = -5; dy < 5; ++dy)
	{
		for (int dx = -5; dx < 5; ++dx)
		{
			int current_x = center_x + dx;
			int current_y = center_y + dy;

			int index = get_index({ current_x, current_y }, my_config.width, my_config.height);
			chem_conc_field[index] = 1000.0;
		}
	}

	return chem_conc_field;
}


