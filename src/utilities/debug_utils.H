#pragma once

#include <iostream>
#include <vector>

namespace openturbine::debug {

void print_array(double* array, int array_len);
void print_vector(std::vector<double> vec);

}  // namespace openturbine::debug
