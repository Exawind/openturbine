
#include "src/utilities/debug_utils.H"

namespace openturbine::debug {

void print_array(double* array, int array_len)
{
    std::cout << "size: " << array_len << std::endl;
    for (int i = 0; i < array_len; i++) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

void print_vector(std::vector<double> vec)
{
    // Borrowed from @Akavall
    // https://stackoverflow.com/questions/27028226/python-linspace-in-c
    std::cout << "size: " << vec.size() << std::endl;
    for (double d : vec) std::cout << d << " ";
    std::cout << std::endl;
}

} // namespace openturbine::debug
