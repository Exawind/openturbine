#pragma once

#include <stdexcept>

#include <KokkosBatched_Gesv.hpp>
#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine::gebt_poc {
inline void solve_linear_system(
    Kokkos::View<double**> A, Kokkos::View<double*> x, Kokkos::View<double*> b
) {
    using member_type = Kokkos::TeamPolicy<>::member_type;
    using no_pivoting = KokkosBatched::Gesv::NoPivoting;
    using gesv = KokkosBatched::TeamVectorGesv<member_type, no_pivoting>;

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

    if (std::isnan(KokkosBlas::sum(x))) {
        throw std::runtime_error("Solution contains NaN values.");
    }
}
}  // namespace openturbine::gebt_poc