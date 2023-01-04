#ifndef HEAT_SOLVE_H
#define HEAT_SOLVE_H

#include <vector>

namespace openturbine::heat_solve {

template<typename T> std::vector<double> linspace(T start_in, T end_in, int num_in);
double* kokkos_laplacian(int axis_size, double* U, double dx);
double* kokkos_1d_heat_conduction(int axis_size, double* U_im1, double dt, double k, double* dU);
double kokkos_calculate_residual(int axis_size, double* U, double* U_im1);

}

#endif /* HEAT_SOLVE_H */