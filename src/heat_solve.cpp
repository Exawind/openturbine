
#include "src/heat_solve.H"

#include <Kokkos_Core.hpp>

#include "src/utilities/debug_utils.H"

namespace openturbine::heat_solve {

double* heat_conduction_solver(int axis_size, double side_length, int n_max, double k, double ic_x0,
                               double ic_x1, double residual_tolerance) {
    // Spatial step size
    double dx = side_length / axis_size;
    // Spatial grid points for the plate
    std::vector<double> axis_points =
        openturbine::heat_solve::linspace(0.0, side_length, axis_size);
    // Artificial time step to drive the solver until heat equilibrium
    double dt = pow(1.0 / axis_size, 2) / (2.0 * k);

    auto U = static_cast<double*>(std::malloc(axis_size * sizeof(double)));
    auto U_im1 = static_cast<double*>(std::malloc(axis_size * sizeof(double)));
    auto deltaU = static_cast<double*>(std::malloc(axis_size * sizeof(double)));

    // Initialize
    Kokkos::parallel_for(
        "U_init", axis_size, KOKKOS_LAMBDA(int i) { U[i] = 0.0; });
    Kokkos::parallel_for(
        "U_im1_init", axis_size, KOKKOS_LAMBDA(int i) { U_im1[i] = 0.0; });
    Kokkos::parallel_for(
        "deltaU_init", axis_size, KOKKOS_LAMBDA(int i) { deltaU[i] = 0.0; });

    // Apply IC
    U[0] = ic_x0;
    U[axis_size - 1] = ic_x1;

    // Solve
    for (int n = 0; n < n_max; n++) {
        // Copy values from U to U at i-1
        for (int i = 0; i < axis_size; i++) U_im1[i] = U[i];

        // TODO: use dx from axis_points to support irregular grid

        deltaU = openturbine::heat_solve::kokkos_laplacian(axis_size, U_im1, dx);
        U = openturbine::heat_solve::kokkos_1d_heat_conduction(axis_size, U_im1, dt, k, deltaU);

        U[0] = U_im1[0];
        U[axis_size - 1] = U_im1[axis_size - 1];

        double residual = openturbine::heat_solve::kokkos_calculate_residual(axis_size, U, U_im1);
        if (residual < residual_tolerance) {
            std::cout << "Converged in " << n << " iterations." << std::endl;
            break;
        }

        // openturbine::debug::print_array(U, axis_size);
    }

    // Free memory
    std::free(U_im1);
    std::free(deltaU);

    return U;
}

template <typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in) {
    // Adapted from @Akavall
    // https://stackoverflow.com/questions/27028226/python-linspace-in-c
    std::vector<double> linspaced;

    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num < 0) throw std::invalid_argument("linspace: num_in must be a positive integer");

    if (num == 0) {
        return linspaced;
    }
    if (num == 1) {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for (int i = 0; i < num - 1; ++i) {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end);
    return linspaced;
}
// Generate the template used in main; alternatively, move this implementation
// the corresponding .H
template std::vector<double> linspace(double, double, int);
// template std::vector<double> linspace(int, int, int);
// template std::vector<double> linspace(int, double, int);
// template std::vector<double> linspace(double, int, int);

double* kokkos_laplacian(int axis_size, double* U, double dx) {
    // Calculate the laplacian for the given 1d array. This does
    // not operate on the boundaries.
    auto laplacian = static_cast<double*>(std::malloc(axis_size * sizeof(double)));
    Kokkos::parallel_for(
        "laplacian", axis_size - 2, KOKKOS_LAMBDA(int i) {
            // std::cout << "Iteration: " << i << std::endl;

            i = i + 1;  // shift over 1 to enforce the BC
            double left = U[i - 1];
            double right = U[i + 1];
            double center = U[i];
            // std::cout << "   " << left << " " << center << " " << right <<
            // std::endl;
            laplacian[i] = (left + right - 2 * center) / pow(dx, 2);
        });
    return laplacian;
}

double* kokkos_1d_heat_conduction(int axis_size, double* U_im1, double dt, double k, double* dU) {
    // Heat conduction model in 1d with no time dependence.
    // NOTE: the boundaries are not computed and are left uninitialized.
    auto heat = static_cast<double*>(std::malloc(axis_size * sizeof(double)));
    Kokkos::parallel_for(
        "1d_heat_conduction", axis_size - 2, KOKKOS_LAMBDA(int i) {
            i = i + 1;  // shift over 1 to enforce the BC
            heat[i] = U_im1[i] + dt * (k * dU[i]);
        });
    return heat;
}

double kokkos_calculate_residual(int axis_size, double* U, double* U_im1) {
    double residual;
    Kokkos::parallel_reduce(
        "residual", axis_size,
        KOKKOS_LAMBDA(const int& i, double& iresidual) { iresidual += U[i] - U_im1[i]; }, residual);
    return residual;
}

}  // namespace openturbine::heat_solve
