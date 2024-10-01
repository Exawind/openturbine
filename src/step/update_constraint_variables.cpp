#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "src/constraints/calculate_constraint_force.hpp"
#include "src/constraints/calculate_constraint_residual_gradient.hpp"
#include "src/constraints/constraints.hpp"
#include "src/state/state.hpp"

namespace openturbine {

void UpdateConstraintVariables(const State& state, Constraints& constraints) {
    auto region = Kokkos::Profiling::ScopedRegion("Update Constraint Variables");

    if (constraints.num == 0) {
        return;
    }

    constraints.UpdateViews();

    Kokkos::parallel_for(
        "CalculateConstraintForce", constraints.num,
        CalculateConstraintForce{
            constraints.type, constraints.target_node_index, constraints.axes, constraints.input,
            state.q, constraints.system_residual_terms}
    );

    Kokkos::parallel_for(
        "CalculateConstraintResidualGradient", constraints.num,
        CalculateConstraintResidualGradient{
            constraints.type, constraints.base_node_index, constraints.target_node_index,
            constraints.X0, constraints.axes, constraints.input, state.q, constraints.residual_terms,
            constraints.base_gradient_terms, constraints.target_gradient_terms}
    );
}

}  // namespace openturbine
