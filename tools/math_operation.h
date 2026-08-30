#pragma once

#include <vector>
#include "SimConfig.h"
#include <random>

int get_index(int_vector2d coordinate_vector, int width, int height);
double discrete_laplacian(const SimConfig& config, int x, int y, const std::vector<double>& field);
void initial_condition(const SimConfig& config, std::vector<double>& chem_con_fig);
double_vector2d discrete_gradient(const SimConfig& config, int x, int y, const std::vector<double>& scalar_field);
void diffusion_calculation(const SimConfig& config, std::vector<double>& chem_conc_field, std::vector<int>& num_cells);