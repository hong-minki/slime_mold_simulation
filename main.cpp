#include <fstream>
#include <iostream>
#include <vector>
#include "tools/SimConfig.h"
#include "tools/wait_closing.h"
#include "tools/math_operation.h"
#include "classes/world_setup.h"
#include "classes/simulation.h"

int main()
{
	constexpr SimConfig my_config
	{ 
		0.1,    // dt
		400,   // total_timesteps
		1.0,    // dx
		100,    // width
		100,    // height
		1.0,	// diffusion_rate
		0.1,    // decay_rate
		1.0,	// chem_secretion_rate

		//Cell properties
		1000,   // total_cells
		2.0,    // chi (Chemotaxis strength)
		0.5     // Dr (Random diffusion) 
	};
	
	std::random_device rd;
	std::mt19937 rng{ rd() };


	class world_setup world(my_config, rng);
	std::vector<double_vector2d> cells_coordinate{ world.random_cell_distribution() };
	std::vector<double> chem_conc_field{ world.chem_conc_field_initialisation_square() };

	class simulation sim(my_config, rng, cells_coordinate, chem_conc_field);
	sim.run_simulation();

	wait_closing();

}	
