#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "constraints/constraints.hpp"
#include "solver/contribute_constraints_system_residual_to_vector.hpp"
#include "solver/contribute_lambda_to_vector.hpp"
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

    Kokkos::parallel_for(
        "ContributeLambdaToVector", constraints.num_constraints,
        ContributeLambdaToVector{
            constraints.base_node_freedom_signature, constraints.target_node_freedom_signature,
            constraints.base_node_freedom_table, constraints.target_node_freedom_table,
            constraints.base_lambda_residual_terms, constraints.target_lambda_residual_terms,
            solver.R
        }
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
