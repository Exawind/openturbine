#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "src/elements/beams/beams.hpp"
#include "src/system/integrate_residual_vector.hpp"

namespace openturbine {

inline void AssembleResidualVector(const Elements& elements) {
    if (elements.ElementType() == ElementsType::kBeams) {
        auto beams = static_cast<const Beams&>(elements);
        auto region = Kokkos::Profiling::ScopedRegion("Assemble Residual");
        auto range_policy = Kokkos::TeamPolicy<>(static_cast<int>(beams.num_elems), Kokkos::AUTO());
        const auto shape_size =
            Kokkos::View<double**>::shmem_size(beams.max_elem_nodes, beams.max_elem_qps);
        const auto weight_size = Kokkos::View<double*>::shmem_size(beams.max_elem_qps);
        const auto node_variable_size = Kokkos::View<double* [6]>::shmem_size(beams.max_elem_nodes);
        const auto qp_variable_size = Kokkos::View<double* [6]>::shmem_size(beams.max_elem_qps);
        // TODO We need to make sure that the scratch space is sufficient for all the views and not
        // use magic numbers here
        range_policy.set_scratch_size(
            1, Kokkos::PerTeam(
                   2 * shape_size + 2 * weight_size + node_variable_size + 5 * qp_variable_size
               )
        );
        Kokkos::parallel_for(
            "IntegrateResidualVector", range_policy,
            IntegrateResidualVector{
                beams.num_nodes_per_element, beams.num_qps_per_element, beams.qp_weight,
                beams.qp_jacobian, beams.shape_interp, beams.shape_deriv, beams.node_FX, beams.qp_Fc,
                beams.qp_Fd, beams.qp_Fi, beams.qp_Fe, beams.qp_Fg, beams.residual_vector_terms
            }
        );

    } else if (elements.ElementType() == ElementsType::kMasses) {
        // For mass elements, numerical integration is not required. Instead, compute the following
        // terms to obtain the full residual vector:
        // 1. Inertial forces (system/CalculateInertialForces)
        // 2. Gravitational forces (system/CalculateGravityForce)
        // 3. External forces (No explicit method for it yet)
        // 4. Constraint forces (constraints/CalculateConstraintForce)
    } else if (elements.ElementType() == ElementsType::kLinearSprings) {
        // TODO: Implement residual vector assembly for linear springs
    } else {
        throw std::invalid_argument("Element type not supported");
    }
}

}  // namespace openturbine
