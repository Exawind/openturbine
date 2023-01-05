
#ifndef DEBUG_UTILS_H
#define DEBUG_UTILS_H

#include <iostream>
#include <vector>

namespace openturbine::debug {

void print_array(double* array, int array_len);
void print_vector(std::vector<double> vec);

} // namespace openturbine::debug

#endif /* DEBUG_UTILS_H */