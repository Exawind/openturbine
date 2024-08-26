#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "solver.hpp"

namespace openturbine {

inline void AssembleConstraintsResidual(Solver& solver) {
    auto resid_region = Kokkos::Profiling::ScopedRegion("Assemble Constraints Residual");

    if (solver.constraints.num == 0) {
        return;
    }

    auto R = Solver::ValuesType(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "R_local"), solver.num_system_dofs
    );
    Kokkos::deep_copy(R, Kokkos::subview(solver.R, Kokkos::make_pair(size_t{0U}, solver.num_system_dofs)));
    auto lambda = Solver::ValuesType(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "lambda"), solver.state.lambda.extent(0)
    );
    Kokkos::deep_copy(lambda, solver.state.lambda);
    auto spmv_handle = Solver::SpmvHandle();
    KokkosSparse::spmv(&spmv_handle, "T", 1., solver.B, lambda, 1., R);
    Kokkos::deep_copy(Kokkos::subview(solver.R, Kokkos::make_pair(size_t{0U}, solver.num_system_dofs)), R);
    Kokkos::deep_copy(Kokkos::subview(solver.R, Kokkos::make_pair(solver.num_system_dofs, solver.num_dofs)), solver.constraints.Phi);
}

}
