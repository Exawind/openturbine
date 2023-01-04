
#include "src/heat_solve.H"
#include <Kokkos_Core.hpp>

namespace openturbine::heat_solve {

template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{
    // Borrowed from @Akavall https://stackoverflow.com/questions/27028226/python-linspace-in-c
    std::vector<double> linspaced;

    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num == 0) { return linspaced; }
    if (num == 1)  {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for(int i=0; i < num-1; ++i) {
        linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end);
    return linspaced;
}
// Generate the template used in main; alternatively, move this implementation the corresponding .H
template std::vector<double> linspace(double, double, int);


double* kokkos_laplacian(int axis_size, double* U, double dx)
{
    // Calculate the laplacian for the given 1d array. This does
    // not operate on the boundaries.
    auto laplacian = static_cast<double*>(std::malloc(axis_size * sizeof(double)));
    Kokkos::parallel_for( "laplacian", axis_size-2, KOKKOS_LAMBDA ( int i ) {
        // std::cout << "Iteration: " << i << std::endl;

        i = i + 1;  // shift over 1 to enforce the BC
        double left = U[i-1];
        double right = U[i+1];
        double center = U[i];
        // std::cout << "   " << left << " " << center << " " << right << std::endl;
        laplacian[i] = (left + right - 2 * center) / pow(dx, 2);
    });
    return laplacian;
}

double* kokkos_1d_heat_conduction(int axis_size, double* U_im1, double dt, double k, double* dU)
{
    // Heat conduction model in 1d with no time dependence.
    // NOTE: the boundaries are not computed and are left uninitialized.
    auto heat = static_cast<double*>(std::malloc(axis_size * sizeof(double)));
    Kokkos::parallel_for( "1d_heat_conduction", axis_size-2, KOKKOS_LAMBDA ( int i ) {
        i = i + 1;  // shift over 1 to enforce the BC
        heat[i] = U_im1[i] + dt * (k * dU[i]);
    });
    return heat;
}

double kokkos_calculate_residual(int axis_size, double* U, double* U_im1)
{
    double residual;
    Kokkos::parallel_reduce("residual", axis_size, KOKKOS_LAMBDA (const int& i, double& iresidual ) {
        iresidual += U[i] - U_im1[i];
    },residual);
    return residual;
}

} // namespace openturbine::heat_solve