#include "math_operation.h"

// Mathematical Tools
int get_index(const int_vector2d coordinate_vector, int width, int height)
{
	int wrapped_x{ (coordinate_vector.x % width + width) % width};
	int wrapped_y{ (coordinate_vector.y % height + height) % height };

	return wrapped_y * width + wrapped_x;
}

double discrete_laplacian(const SimConfig& config, const int_vector2d coordinate_vector, const std::vector<double>& field)
{
	int center_index{ get_index(coordinate_vector, config.width, config.height) };
	int left_index{ get_index({ coordinate_vector.x - 1, coordinate_vector.y }, config.width, config.height) };
	int right_index{ get_index({ coordinate_vector.x + 1, coordinate_vector.y }, config.width, config.height) };
	int up_index{ get_index({ coordinate_vector.x, coordinate_vector.y - 1 }, config.width, config.height) };
	int down_index{ get_index({ coordinate_vector.x, coordinate_vector.y + 1 }, config.width, config.height) };

	double laplacian = (field[left_index] + field[right_index] + field[up_index] + field[down_index] - 4 * field[center_index]) / (config.dx * config.dx);

	return laplacian;
}

double_vector2d discrete_gradient(const SimConfig& config, const int_vector2d coordinate_vector, const std::vector<double>& scalar_field)
{
	int left_index{ get_index({ coordinate_vector.x - 1, coordinate_vector.y }, config.width, config.height) };
	int right_index{ get_index({ coordinate_vector.x + 1, coordinate_vector.y }, config.width, config.height) };
	int up_index{ get_index({ coordinate_vector.x, coordinate_vector.y - 1 }, config.width, config.height) };
	int down_index{ get_index({ coordinate_vector.x, coordinate_vector.y + 1 }, config.width, config.height) };

	double gradient_x = (scalar_field[right_index] - scalar_field[left_index]) / (2 * config.dx);
	double gradient_y = (scalar_field[down_index] - scalar_field[up_index]) / (2 * config.dx);

	return { gradient_x, gradient_y };
}

//Simulation Tools

void diffusion_calculation(const SimConfig& config, std::vector<double>& chem_conc_field, std::vector<int>& num_cells)
{
	std::vector<double> next_chem_conc_field = chem_conc_field;

	for (int y = 0; y < config.height; ++y)
	{
		for (int x = 0; x < config.width; ++x)
		{
			int_vector2d coordinate_vector = { x, y };

			int center_index = get_index(coordinate_vector, config.width, config.height);
			double chem_conc_new = chem_conc_field[center_index] + config.dt * (config.diffusion_rate * discrete_laplacian(config, coordinate_vector, chem_conc_field)
				- config.decay_rate * chem_conc_field[center_index]
				+ config.chem_secretion_rate * num_cells[center_index]);
			if (chem_conc_new < 0.0) chem_conc_new = 0.0;
			next_chem_conc_field[center_index] = chem_conc_new;
		}
	}
	chem_conc_field.swap(next_chem_conc_field);
}
