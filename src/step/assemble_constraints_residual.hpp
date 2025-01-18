#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "constraints/constraints.hpp"
#include "solver/contribute_constraints_system_residual_to_vector.hpp"
#include "solver/copy_constraints_residual_to_vector.hpp"
#include "solver/solver.hpp"

namespace openturbine {

inline void AssembleConstraintsResidual(Solver& solver, Constraints& constraints) {
    auto resid_region = Kokkos::Profiling::ScopedRegion("Assemble Constraints Residual");

    if (constraints.num_constraints == 0) {
        return;
    }

    Kokkos::parallel_for(
        "ContributeConstraintsSystemResidualToVector", constraints.num_constraints,
        ContributeConstraintsSystemResidualToVector{
            constraints.target_node_freedom_table, constraints.system_residual_terms, solver.R
        }
    );

    using CrsMatrixType = Solver::CrsMatrixType;
    using VectorType = CrsMatrixType::values_type::non_const_type;

    auto R = VectorType(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "R_local"), solver.num_system_dofs
    );
    Kokkos::deep_copy(
        R, Kokkos::subview(solver.R, Kokkos::make_pair(size_t{0U}, solver.num_system_dofs))
    );
    auto lambda = VectorType(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "lambda"), constraints.lambda.extent(0)
    );
    Kokkos::deep_copy(lambda, constraints.lambda);

    auto spmv_handle =
        KokkosSparse::SPMVHandle<Solver::ExecutionSpace, CrsMatrixType, VectorType, VectorType>();
    KokkosSparse::spmv(&spmv_handle, "T", 1., solver.B, lambda, 1., R);
    Kokkos::deep_copy(
        Kokkos::subview(solver.R, Kokkos::make_pair(size_t{0U}, solver.num_system_dofs)), R
    );

    Kokkos::parallel_for(
        "CopyConstraintsResidualToVector", constraints.num_constraints,
        CopyConstraintsResidualToVector{
            constraints.row_range,
            Kokkos::subview(solver.R, Kokkos::make_pair(solver.num_system_dofs, solver.num_dofs)),
            constraints.residual_terms
        }
    );
}

}  // namespace openturbine
