#pragma once

#include <vector>
#include "SimConfig.h"
#include <random>

int get_index(const int_vector2d coordinate_vector, int width, int height);

inline double discrete_laplacian(const SimConfig& config, const auto coordinate_vector, const std::vector<double>& field)
{

	int x_int{ static_cast<int>(coordinate_vector.x) };
	int y_int{ static_cast<int>(coordinate_vector.y) };

	int center_index{ get_index({x_int, y_int}, config.width, config.height) };
	int left_index{ get_index({ x_int - 1, y_int }, config.width, config.height) };
	int right_index{ get_index({ x_int + 1, y_int }, config.width, config.height) };
	int up_index{ get_index({ x_int, y_int - 1 }, config.width, config.height) };
	int down_index{ get_index({ x_int, y_int + 1 }, config.width, config.height) };

	double laplacian = (field[left_index] + field[right_index] + field[up_index] + field[down_index] - 4.0 * field[center_index]) / (config.dx * config.dx);

	return laplacian;
}

inline double_vector2d discrete_gradient(const SimConfig& config, const auto coordinate_vector, const std::vector<double>& scalar_field)
{
	int x_int{ static_cast<int>(coordinate_vector.x) };
	int y_int{ static_cast<int>(coordinate_vector.y) };

	int left_index{ get_index({ x_int - 1, y_int }, config.width, config.height) };
	int right_index{ get_index({ x_int + 1, y_int }, config.width, config.height) };
	int up_index{ get_index({ x_int, y_int - 1 }, config.width, config.height) };
	int down_index{ get_index({ x_int, y_int + 1 }, config.width, config.height) };

	double gradient_x = (scalar_field[right_index] - scalar_field[left_index]) / (2.0 * config.dx);
	double gradient_y = (scalar_field[down_index] - scalar_field[up_index]) / (2.0 * config.dx);

	return { gradient_x, gradient_y };
}