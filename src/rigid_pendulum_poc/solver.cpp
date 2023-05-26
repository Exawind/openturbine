#include "src/rigid_pendulum_poc/solver.h"

#include <lapacke.h>

namespace openturbine::rigid_pendulum {

int solve_linear_system(Kokkos::View<double**> system, Kokkos::View<double*> solution) {
    auto rows = static_cast<int>(system.extent(0));
    auto columns = static_cast<int>(system.extent(1));

    if (rows != columns) {
        throw std::invalid_argument("Provided system must be a square matrix");
    }

    if (rows != static_cast<int>(solution.extent(0))) {
        throw std::invalid_argument(
            "Provided system and solution must contain the same number of rows");
    }

    int right_hand_sides{1};
    int leading_dimension_sytem{rows};
    auto pivots = Kokkos::View<int*, Kokkos::DefaultHostExecutionSpace>("pivots", solution.size());
    int leading_dimension_solution{1};

    // Call DGESV from LAPACK to compute the solution to a real system of linear
    // equations A * x = b, returns 0 if successful
    // https://www.netlib.org/lapack/lapacke.html
    return LAPACKE_dgesv(LAPACK_ROW_MAJOR,  // input: matrix_layout
                         rows,              // input: number of linear equations
                         right_hand_sides,  // input: number of rhs
                         system.data(),     // input/output: Upon entry, the nxn coefficient matrix
                                            // Upon exit, the factors L and U from the factorization
                         leading_dimension_sytem,  // input: leading dimension of system
                         pivots.data(),            // output: pivot indices
                         solution.data(),  // input/output: Upon entry, the right-hand side matrix
                                           // Upon exit, the solution matrix
                         leading_dimension_solution  // input: leading dimension of solution
    );
}

}  // namespace openturbine::rigid_pendulum
