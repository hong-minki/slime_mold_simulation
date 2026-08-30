# CMakeList.txt : CMake project for slime_mold_simulation, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (slime_mold_simulation "main.cpp" "main.h" "tools/wait_closing.cpp" "tools/math_operation.cpp" "tools/math_operation.h" "tools/SimConfig.h" "tools/wait_closing.h" "classes/world_setup.cpp" "classes/world_setup.h" "classes/simulation.cpp" "classes/simulation.h")

