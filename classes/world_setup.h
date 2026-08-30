#pragma once
#include <random> 
#include <vector>
#include "../tools/SimConfig.h"
#include "../tools/math_operation.h"

class world_setup
{
private:
	const SimConfig& my_config;
	
	// 2. Declare random number generation tools
	std::random_device rd;
	std::mt19937 rng;
	std::uniform_real_distribution<double> dist_x;
	std::uniform_real_distribution<double> dist_y;


public:
	world_setup(const SimConfig& incoming_config);

	std::vector<double_vector2d> random_cell_distribution();
	std::vector<double> chem_conc_field_initialisation_square();
};