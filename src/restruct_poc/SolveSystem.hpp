#pragma once

#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
#include <KokkosLapack_gesv.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "solver.hpp"
#include "ConditionSystem.hpp"

namespace openturbine {

inline void SolveSystem(Solver& solver) {
    auto region = Kokkos::Profiling::ScopedRegion("Solve System");
    Kokkos::parallel_for(
        "PreconditionSt",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {solver.num_system_dofs, solver.num_dofs}),
        PreconditionSt{
            solver.St,
            solver.conditioner,
        }
    );
    Kokkos::parallel_for(
        "PostconditionSt",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, solver.num_system_dofs}, {solver.num_dofs, solver.num_dofs}
        ),
        PostconditionSt{
            solver.St,
            solver.conditioner,
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
    Kokkos::deep_copy(solver.St_left, solver.St);
    auto x = Kokkos::View<double*, Kokkos::LayoutLeft>(solver.x);
    KokkosLapack::gesv(solver.St_left, x, solver.IPIV);
    Kokkos::deep_copy(solver.x, x);

    Kokkos::parallel_for(
        "UnconditionSolution", solver.num_dofs,
        UnconditionSolution{
            solver.num_system_dofs,
            solver.conditioner,
            solver.x,
        }
    );
}

}
