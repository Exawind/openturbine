#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"

#include "src/math/quaternion_operations.hpp"
#include "src/math/vector_operations.hpp"
#include "src/types.hpp"

namespace openturbine {

struct CalculateRevoluteJointForce {
    int i_constraint;
    Kokkos::View<size_t*>::const_type target_node_index;
    Kokkos::View<double* [3][3]>::const_type axes;
    Kokkos::View<double* [7]>::const_type inputs;
    Kokkos::View<double* [7]>::const_type node_u;
    Kokkos::View<double* [6]> system_residual_terms;

    KOKKOS_FUNCTION
    void operator()() const {
        const auto i_node2 = target_node_index(i_constraint);

        // Initial difference between nodes
        const auto X0_data = Kokkos::Array<double, 3>{
            axes(i_constraint, 0, 0), axes(i_constraint, 0, 1), axes(i_constraint, 0, 2)
        };
        const auto X0 = View_3::const_type{X0_data.data()};

        auto R2_X0_data = Kokkos::Array<double, 3>{};
        const auto R2_X0 = Kokkos::View<double[3]>{R2_X0_data.data()};

        // Target node displacement
        const auto R2_data = Kokkos::Array<double, 4>{
            node_u(i_node2, 3), node_u(i_node2, 4), node_u(i_node2, 5), node_u(i_node2, 6)
        };
        const auto R2 = Kokkos::View<double[4]>::const_type{R2_data.data()};

        //----------------------------------------------------------------------
        // Residual Vector
        //----------------------------------------------------------------------

        // Extract residual rows relevant to this constraint
        RotateVectorByQuaternion(R2, X0, R2_X0);

        // Take axis_x and rotate it to right orientation
        system_residual_terms(i_constraint, 3) = R2_X0(0) * inputs(i_constraint, 0);
        system_residual_terms(i_constraint, 4) = R2_X0(1) * inputs(i_constraint, 0);
        system_residual_terms(i_constraint, 5) = R2_X0(2) * inputs(i_constraint, 0);
    }
};

}  // namespace openturbine
