#include "src/utilities/debug_utils.H"

#include "src/utilities/log.h"

namespace openturbine::debug {

void print_array(double* array, int array_len) {
    auto log = Log::Get();
    log->Info("array size: " + std::to_string(array_len) + "\n");

    std::string array_to_print {"array elements:"};
    for (int i = 0; i < array_len; i++) {
        array_to_print += " " + std::to_string(array[i]);
    }
    log->Info(array_to_print + "\n");
}

void print_vector(std::vector<double> vec) {
    // Borrowed from @Akavall
    // https://stackoverflow.com/questions/27028226/python-linspace-in-c
    auto log = Log::Get();
    log->Info("vector size: " + std::to_string(vec.size()) + "\n");

    std::string vector_to_print {"vector elements:"};
    for (double d : vec) {
        vector_to_print += " " + std::to_string(d);
    }
    log->Info(vector_to_print + "\n");
}

}  // namespace openturbine::debug
