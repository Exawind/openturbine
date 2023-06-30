#pragma once

#include <vector>

namespace openturbine::heat_solve {

double* heat_conduction_solver(
    int axis_size, double side_length, int n_max, double k, double ic_x0, double ic_x1,
    double residual_tolerance
);

template <typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in);

double* kokkos_laplacian(int axis_size, double* U, double dx);

double* kokkos_1d_heat_conduction(int axis_size, double* U_im1, double dt, double k, double* dU);

double kokkos_calculate_residual(int axis_size, double* U, double* U_im1);

}  // namespace openturbine::heat_solve
