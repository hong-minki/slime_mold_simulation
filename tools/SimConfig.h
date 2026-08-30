#pragma once

struct SimConfig {
    double dt;
    int total_timesteps;
    double dx;
    int width;
    int height;
    double diffusion_rate;
    double decay_rate;
    double chem_secretion_rate;
    
    //Cell properties
    int total_cells;
    double chi;
    double Dr;

    int total_grid_points() const {
        return width * height;
    }
};

struct int_vector2d
{
    int x;
    int y;
};

struct double_vector2d
{
    double x;
    double y;
};