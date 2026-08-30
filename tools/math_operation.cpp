#include "math_operation.h"

// Mathematical Tools
int get_index(const int_vector2d coordinates_vector, int width, int height)
{
	int wrapped_x{ (coordinates_vector.x % width + width) % width};
	int wrapped_y{ (coordinates_vector.y % height + height) % height };

	return wrapped_y * width + wrapped_x;
}



//Simulation Tools
