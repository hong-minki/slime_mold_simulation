#include "wait_closing.h"
#include <iostream>


void wait_closing()
{
    std::cout << "\nPress Enter to close this window...";
    std::cin.clear(); // Clear any input errors
    std::cin.ignore(10000, '\n'); // Clear the leftover data in the buffer
    std::cin.get(); // Wait for the user to press Enter
}