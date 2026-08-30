#include "math_operation.h"

// Mathematical Tools
int get_index(const int_vector2d coordinate_vector, int width, int height)
{
	int wrapped_x{ (coordinate_vector.x % width + width) % width};
	int wrapped_y{ (coordinate_vector.y % height + height) % height };

	return wrapped_y * width + wrapped_x;
}



//Simulation Tools
