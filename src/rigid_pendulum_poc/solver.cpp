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
    int leading_dimension{rows};

    // Array of pivot indices used for the LU factorization of system
    auto pivots = Kokkos::View<int*, Kokkos::DefaultHostExecutionSpace>("pivots", solution.size());

    return LAPACKE_dgesv(LAPACK_ROW_MAJOR, rows, right_hand_sides, system.data(), leading_dimension,
                         pivots.data(), solution.data(), 1);
}

}  // namespace openturbine::rigid_pendulum
