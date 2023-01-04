
#include "src/heat_solve.H"
#include "src/utilities/console_io.H"
#include "src/utilities/debug_utils.H"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <Kokkos_Core.hpp>


int main( int argc, char* argv[] )
{

    if (argc > 2) {
        // Print usage and exit with error code if no input file was provided.
        openturbine::io::print_usage(std::cout);
        openturbine::io::print_error("No input file provided. Exiting.");
        return 1;
    }

    // cppcheck-suppress knownConditionTrueFalse
    // Look for "-h" or "--help" flag and print usage
    for (auto i = 1; i < argc; i++) {
        const std::string param(argv[i]);
        if ((param == "--help") || (param == "-h")) {
            openturbine::io::print_banner(std::cout);
            openturbine::io::print_usage(std::cout);
            return 0;
        }
    }

    int axis_size = 10;                // Size of the 1D grid
    double side_length = 1.0;          // Length of the 1D domain
    double k = 1.011;                  // Thermal diffusivity of the material in units of cm^2/s
    double ic_x0 = 100.0;              // Initial temperature at the beginning of the domain
    double ic_x1 = 100.0;              // Initial temperature at the end of the domain
    int n_max = 500;                   // Max iterations
    double residual_tolerance = 1e-5;  // Iterative residual tolerance
    
    auto U = static_cast<double*>(std::malloc(axis_size * sizeof(double)));

    Kokkos::initialize( argc, argv );
    openturbine::heat_solve::heat_conduction_solver(
        axis_size,
        side_length,
        n_max,
        k,
        ic_x0,
        ic_x1,
        residual_tolerance
    );
    Kokkos::finalize();

    openturbine::debug::print_array(U, axis_size);

    std::free(U);

    return 0;
}
