#pragma once

#include <vector>
#include "SimConfig.h"
#include <random>

int get_index(int_vector2d coordinate_vector, int width, int height);
double discrete_laplacian(const SimConfig& config, int_vector2d coordinate_vector, const std::vector<double>& field);
double_vector2d discrete_gradient(const SimConfig& config, int_vector2d coordinate_vector, const std::vector<double>& scalar_field);
