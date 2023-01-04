
#include "src/heat_solve.H"
#include "src/utilities/console_io.H"
#include "src/utilities/debug_utils.H"

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

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

    int axis_size = 10;                                                         // Size of the 1D grid
    double side_length = 1.0;                                                   // Length of the 1D domain
    double dx = side_length/axis_size;                                          // Spatial step size
    std::vector<double> axis_points = 
        openturbine::heat_solve::linspace(0.0, side_length, axis_size);         // Spatial grid points for the plate
    double k = 1.011;                                                           // Thermal diffusivity of the material in units of cm^2/s
    int n_max = 500;                                                            // Max iterations
    double dt = pow( 1.0 / axis_size, 2) / (2.0 * k);                           // Artificial time step to drive the solver until heat equilibrium
    double residual_tolerance = 1e-5;                                           // Iterative residual tolerance

    Kokkos::initialize( argc, argv );

    {
        auto U = static_cast<double*>(std::malloc(axis_size * sizeof(double)));
        auto U_im1 = static_cast<double*>(std::malloc(axis_size * sizeof(double)));
        auto deltaU = static_cast<double*>(std::malloc(axis_size * sizeof(double)));
        auto residual = static_cast<double*>(std::malloc(axis_size * sizeof(double)));

        // Initialize
        Kokkos::parallel_for( "U_init", axis_size, KOKKOS_LAMBDA ( int i ) {
            U[ i ] = 0.0;
        });
        Kokkos::parallel_for( "U_im1_init", axis_size, KOKKOS_LAMBDA ( int i ) {
            U_im1[ i ] = 0.0;
        });
        Kokkos::parallel_for( "deltaU_init", axis_size, KOKKOS_LAMBDA ( int i ) {
            deltaU[ i ] = 0.0;
        });
        Kokkos::parallel_for( "deltaU_init", axis_size, KOKKOS_LAMBDA ( int i ) {
            residual[ i ] = 0.0;
        });

        // Apply IC
        U[0] = 100.0;
        U[axis_size-1] = 100.0;

        // Solve
        for ( int n = 0; n < n_max; n++ ) {

            // Copy values from U to U at i-1
            for ( int i = 0; i < axis_size; i++) U_im1[i] = U[i];

            deltaU = openturbine::heat_solve::kokkos_laplacian(axis_size, U_im1, dx);
            U = openturbine::heat_solve::kokkos_1d_heat_conduction(axis_size, U_im1, dt, k, deltaU);

            U[0] = U_im1[0];
            U[axis_size-1] = U_im1[axis_size-1];

            double residual = openturbine::heat_solve::kokkos_calculate_residual(axis_size, U, U_im1);
            if (residual < residual_tolerance)
            {
                std::cout << "Converged in " << n << " iterations." << std::endl;
                break;
            }

            openturbine::debug::print_array(U, axis_size);
        }

        // Free memory
        std::free(U);
        std::free(U_im1);
        std::free(deltaU);
        std::free(residual);
    }

    Kokkos::finalize();

    return 0;
}
