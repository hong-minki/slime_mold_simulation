#pragma once
#include "../tools/SimConfig.h"
#include "../tools/math_operation.h"
#include <fstream>
#include <iostream>
#include <random> 

class simulation
{
private:
	const SimConfig& my_config;
	std::mt19937& rng;
	std::vector<double_vector2d>& cells_coordinates;
	std::vector<double>& chem_conc_field;
	std::vector<int> num_cells; 

	void cells_update();	
	void diffusion_calculation();

public:
	simulation(const SimConfig& incoming_config, std::mt19937& incoming_rng, std::vector<double_vector2d>& incoming_cells_coordinates, std::vector<double>& incoming_chem_conc_field);
	void run_simulation();
};