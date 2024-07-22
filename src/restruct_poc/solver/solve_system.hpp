#pragma once

#include <KokkosBlas.hpp>
#include <KokkosLapack_gesv.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "condition_system.hpp"
#include "solver.hpp"

namespace openturbine {

inline void SolveSystem(Solver& solver) {
    auto region = Kokkos::Profiling::ScopedRegion("Solve System");

    {
        auto assemble_region = Kokkos::Profiling::ScopedRegion("Assemble Full System");
        KokkosSparse::spadd_numeric(
            &solver.spc_spadd_handle, solver.conditioner, solver.system_matrix_full, 1.,
            solver.constraints_matrix_full, solver.system_plus_constraints
        );
        KokkosSparse::spadd_numeric(
            &solver.full_system_spadd_handle, 1., solver.system_plus_constraints, 1.,
            solver.transpose_matrix_full, solver.full_matrix
        );
    }

    auto St = solver.St;
    auto full_matrix = solver.full_matrix;
    auto sparse_matrix_policy = Kokkos::TeamPolicy<>(full_matrix.numRows(), Kokkos::AUTO());
    Kokkos::parallel_for(
        "Copy into St", sparse_matrix_policy,
        KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type member) {
            auto i = member.league_rank();
            auto row = full_matrix.row(i);
            auto row_map = full_matrix.graph.row_map;
            auto cols = full_matrix.graph.entries;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, row.length), [=](int entry) {
                St(i, cols(row_map(i) + entry)) = row.value(entry);
            });
        }
    );

    Kokkos::parallel_for(
        "ConditionR", solver.num_system_dofs,
        ConditionR{
            solver.R,
            solver.conditioner,
        }
    );

    KokkosBlas::axpby(-1.0, solver.R, 0.0, solver.x);
    auto x = Kokkos::View<double*, Kokkos::LayoutLeft>(solver.x);
    if constexpr (std::is_same_v<decltype(solver.St)::array_layout, Kokkos::LayoutLeft>) {
        KokkosLapack::gesv(solver.St, x, solver.IPIV);
    } else {
        Kokkos::deep_copy(solver.St_left, solver.St);
        KokkosLapack::gesv(solver.St_left, x, solver.IPIV);
    }

    Kokkos::parallel_for(
        "UnconditionSolution", solver.num_dofs,
        UnconditionSolution{
            solver.num_system_dofs,
            solver.conditioner,
            solver.x,
        }
    );
}

}  // namespace openturbine
