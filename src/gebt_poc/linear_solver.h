#pragma once

#include <stdexcept>

#include <KokkosBlas.hpp>

#include "KokkosLapack_gesv.hpp"

#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

static constexpr double kConvergenceTolerance = 1e-12;

inline void solve_linear_system(Kokkos::View<double**> system, Kokkos::View<double*> solution) {
    auto A =
        Kokkos::View<double**, Kokkos::LayoutLeft>("system", system.extent(0), system.extent(1));
    Kokkos::deep_copy(A, system);
    auto A_host = Kokkos::create_mirror(A);
    Kokkos::deep_copy(A_host, A);
    auto b = Kokkos::View<double*, Kokkos::LayoutLeft>("solution", solution.extent(0));
    Kokkos::deep_copy(b, solution);
    auto b_host = Kokkos::create_mirror(b);
    Kokkos::deep_copy(b_host, b);
    auto pivots =
        Kokkos::View<int*, Kokkos::LayoutLeft, Kokkos::HostSpace>("pivots", solution.extent(0));

    KokkosLapack::gesv(A_host, b_host, pivots);
    Kokkos::deep_copy(A, A_host);
    Kokkos::deep_copy(b, b_host);

    // Here we add some safety checks to make sure the returned solution and/or the provided
    // linear system is valid
    // Check 1: Check for NaN values in the solution and throw if found
    if (std::isnan(KokkosBlas::sum(b))) {
        throw std::runtime_error("Solution contains NaN values.");
    }

    // Check 2: Check [A] * {x} - {b} = {0} and throw if not
    auto size = solution.extent(0);
    auto residual = Kokkos::View<double*>("residual", size);
    Kokkos::deep_copy(residual, solution);
    KokkosBlas::gemv("N", 1., system, b, -1., residual);
    auto norm = KokkosBlas::nrm2(residual) / size;
    if (norm > kConvergenceTolerance) {
        auto log = util::Log::Get();
        log->Error(
            "Residual norm i.e. [A]*{x}-{b} exceeds tolerance: " + std::to_string(norm) + "\n"
        );
        throw std::runtime_error("Linear system solver failed to find a solution");
    }

    Kokkos::deep_copy(system, A);
    Kokkos::deep_copy(solution, b);
}

}  // namespace openturbine::gebt_poc
