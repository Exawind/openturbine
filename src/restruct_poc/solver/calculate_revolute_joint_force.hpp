#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"

#include "src/restruct_poc/math/quaternion_operations.hpp"
#include "src/restruct_poc/math/vector_operations.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct CalculateRevoluteJointForce {
    Kokkos::View<Constraints::DeviceData*>::const_type data;
    View_N::const_type control;
    View_Nx7::const_type node_u;
    View_N R_system;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        const auto& cd = data(i_constraint);
        const auto i_node2 = cd.target_node_index;

        // Initial difference between nodes
        const auto X0_data = Kokkos::Array<double, 3>{cd.axis_x[0], cd.axis_x[1], cd.axis_x[2]};
        const auto X0 = View_3::const_type{X0_data.data()};

        auto R2_X0_data = Kokkos::Array<double, 3>{};
        const auto R2_X0 = Kokkos::View<double[3]>{R2_X0_data.data()};

        // Target node displacement
        const auto R2_data = Kokkos::Array<double, 4>{
            node_u(i_node2, 3), node_u(i_node2, 4), node_u(i_node2, 5), node_u(i_node2, 6)};
        const auto R2 = Kokkos::View<double[4]>::const_type{R2_data.data()};

        //----------------------------------------------------------------------
        // Residual Vector
        //----------------------------------------------------------------------

        // Extract residual rows relevant to this constraint
        auto R = Kokkos::subview(
            R_system,
            Kokkos::make_pair(i_node2 * kLieGroupComponents, (i_node2 + 1) * kLieGroupComponents)
        );
        RotateVectorByQuaternion(R2, X0, R2_X0);

        // Take axis_x and rotate it to right orientation
        R(3) += R2_X0(0) * control(i_constraint);
        R(4) += R2_X0(1) * control(i_constraint);
        R(5) += R2_X0(2) * control(i_constraint);
    }
};

}  // namespace openturbine
