#pragma once

#include <stdexcept>

#include <KokkosBatched_QR_WithColumnPivoting_Decl.hpp>
#include <KokkosBatched_SolveUTV_Decl.hpp>
#include <KokkosBatched_UTV_Decl.hpp>
#include <KokkosBlas.hpp>
#include <KokkosLapack_gesv.hpp>

#include "src/utilities/log.h"

namespace openturbine::gebt_poc {

static constexpr double kConvergenceTolerance = 1e-12;

inline void solve_linear_system(Kokkos::View<double**> system, Kokkos::View<double*> solution) {
    // Define the complex templated types used in the solver
    using member_type = Kokkos::TeamPolicy<>::member_type;
    using decompose = KokkosBatched::TeamVectorUTV<member_type, KokkosBatched::Algo::UTV::Unblocked>;
    using solve =
        KokkosBatched::TeamVectorSolveUTV<member_type, KokkosBatched::Algo::UTV::Unblocked>;
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using unmanaged_memory = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using matrix = Kokkos::View<double**, scratch_space, unmanaged_memory>;
    using vector = Kokkos::View<double*, scratch_space, unmanaged_memory>;
    using scalar_vector = Kokkos::View<int*, scratch_space, unmanaged_memory>;

    // Create local copies of the system and solution to be used in the solver
    auto A = Kokkos::View<double**>("A", system.extent(0), system.extent(1));
    Kokkos::deep_copy(A, system);
    auto x = Kokkos::View<double*>("x", solution.extent(0));
    auto b = Kokkos::View<double*>("b", solution.extent(0));
    Kokkos::deep_copy(b, solution);

    // Setup the scratch space needed for the solver
    auto policy = Kokkos::TeamPolicy<>(1, Kokkos::AUTO(), Kokkos::AUTO());
    auto n = A.extent(0);
    auto scratch_size =
        2 * matrix::shmem_size(n, n) + vector::shmem_size(3 * n) + scalar_vector::shmem_size(n);
    policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));

    // Solve the system using UTV decomposition
    Kokkos::parallel_for(
        policy,
        KOKKOS_LAMBDA(const member_type& member) {
            int rank = 0;
            auto U = matrix(member.team_scratch(0), n, n);
            auto T = A;
            auto V = matrix(member.team_scratch(0), n, n);
            auto p = scalar_vector(member.team_scratch(0), n);
            auto w = vector(member.team_scratch(0), 3 * n);

            if (decompose::invoke(member, A, p, U, V, w, rank)) {
                Kokkos::abort("Decomposition Failed");
            }
            member.team_barrier();
            if (solve::invoke(member, rank, U, T, V, p, x, b, w)) {
                Kokkos::abort("Solve Failed");
            }
            member.team_barrier();
        }
    );
    Kokkos::fence();

    // Copy the solution into b to maintain consistency with debug checks
    Kokkos::deep_copy(b, x);

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
