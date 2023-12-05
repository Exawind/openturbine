#include "src/gen_alpha_poc/solver.h"

#include "src/gebt_poc/linear_solver.h"

namespace openturbine::gen_alpha_solver {

void solve_linear_system(Kokkos::View<double**> system, Kokkos::View<double*> solution) {
    auto rhs = Kokkos::View<double*>("rhs", solution.extent(0));
    Kokkos::deep_copy(rhs, solution);

    openturbine::gebt_poc::solve_linear_system(system, solution, rhs);
}

}  // namespace openturbine::gen_alpha_solver
