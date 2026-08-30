set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable (slime_mold_simulation "main.cpp" "main.h" "tools/wait_closing.cpp" "tools/math_operation.cpp" "tools/math_operation.h" "tools/SimConfig.h" "tools/wait_closing.h" "classes/world_setup.cpp" "classes/world_setup.h" "classes/simulation.cpp" "classes/simulation.h")

