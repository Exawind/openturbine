#pragma once

#include <stdexcept>

#include <KokkosBatched_Gesv.hpp>
#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

static constexpr double kConvergenceTolerance = 1e-12;

inline void solve_linear_system(
    Kokkos::View<double**> A, Kokkos::View<double*> x, Kokkos::View<double*> b
) {
    using member_type = Kokkos::TeamPolicy<>::member_type;
    using no_pivoting = KokkosBatched::Gesv::NoPivoting;
    using gesv = KokkosBatched::TeamVectorGesv<member_type, no_pivoting>;

    auto system = Kokkos::View<double**>("system", A.extent(0), A.extent(1));
    Kokkos::deep_copy(system, A);

    auto policy = Kokkos::TeamPolicy<>(1, Kokkos::AUTO(), Kokkos::AUTO());
    auto n = A.extent(0);
    auto scratch_size = decltype(A)::shmem_size(n, n + 4);
    policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

    Kokkos::parallel_for(
        policy,
        KOKKOS_LAMBDA(const member_type& member) {
            auto status = gesv::invoke(member, A, x, b);
            member.team_barrier();
            if (status) {
                Kokkos::abort("Linear system solve failed");
            }
        }
    );
    Kokkos::fence();

    // Here add some checks to make sure the solution is valid
    // Check 1: Check for NaN values in the solution and throw if found
    if (std::isnan(KokkosBlas::sum(x))) {
        throw std::runtime_error("Solution contains NaN values.");
    }
    // Check 2: Check [A] * {x} - {b} = {0} and throw if not
    auto size = x.extent(0);
    auto residual = Kokkos::View<double*>("residual", size);
    Kokkos::deep_copy(residual, b);
    KokkosBlas::gemv("N", 1., system, x, -1., residual);
    auto norm = KokkosBlas::nrm2(residual) / size;
    if (norm > kConvergenceTolerance) {
        auto log = util::Log::Get();
        log->Error(
            "Residual norm i.e. [A]*{x}-{b} exceeds tolerance: " + std::to_string(norm) + "\n"
        );
        throw std::runtime_error("Linear system solver failed to find a solution");
    }
}

}  // namespace openturbine::gebt_poc
