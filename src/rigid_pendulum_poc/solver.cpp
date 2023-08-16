#include "src/rigid_pendulum_poc/solver.h"

#include <KokkosBlas_gesv.hpp>

#include "src/utilities/log.h"

namespace openturbine::rigid_pendulum {

void solve_linear_system(Kokkos::View<double**> system, Kokkos::View<double*> solution) {
    auto A = Kokkos::View<double**, Kokkos::LayoutLeft>("system", system.extent(0), system.extent(1));
    Kokkos::deep_copy(A, system);
    auto b = Kokkos::View<double*, Kokkos::LayoutLeft>("solution", solution.extent(0));
    Kokkos::deep_copy(b, solution);
    auto pivots = Kokkos::View<int*, Kokkos::LayoutLeft, Kokkos::HostSpace>("pivots", solution.extent(0));

    KokkosBlas::gesv(A, b, pivots);

    Kokkos::deep_copy(system, A);
    Kokkos::deep_copy(solution, b);
}

}  // namespace openturbine::rigid_pendulum
