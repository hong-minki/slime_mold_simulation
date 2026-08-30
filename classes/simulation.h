#pragma once
#include "../tools/SimConfig.h"
#include "../tools/math_operation.h"
#include <fstream>
#include <iostream>

class simulation
{
private:
	const SimConfig& my_config;
	std::vector<double_vector2d>& cells_coordinate;
	std::vector<double>& chem_conc_field;

public:
	simulation(const SimConfig& incoming_config, std::vector<double_vector2d>& incoming_cells_coordinate, std::vector<double>& incoming_chem_conc_field);
	void run_simulation();
};