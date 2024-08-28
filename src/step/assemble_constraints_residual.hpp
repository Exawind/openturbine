#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "src/constraints/constraints.hpp"
#include "src/solver/solver.hpp"

namespace openturbine {

inline void AssembleConstraintsResidual(Solver& solver, Constraints& constraints) {
    auto resid_region = Kokkos::Profiling::ScopedRegion("Assemble Constraints Residual");

    if (constraints.num == 0) {
        return;
    }

    auto R = Solver::ValuesType(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "R_local"), solver.num_system_dofs
    );
    Kokkos::deep_copy(
        R, Kokkos::subview(solver.R, Kokkos::make_pair(size_t{0U}, solver.num_system_dofs))
    );
    auto lambda = Solver::ValuesType(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "lambda"), constraints.lambda.extent(0)
    );
    Kokkos::deep_copy(lambda, constraints.lambda);
    auto spmv_handle = Solver::SpmvHandle();
    KokkosSparse::spmv(&spmv_handle, "T", 1., solver.B, lambda, 1., R);
    Kokkos::deep_copy(
        Kokkos::subview(solver.R, Kokkos::make_pair(size_t{0U}, solver.num_system_dofs)), R
    );
    Kokkos::deep_copy(
        Kokkos::subview(solver.R, Kokkos::make_pair(solver.num_system_dofs, solver.num_dofs)),
        constraints.Phi
    );
}

}  // namespace openturbine