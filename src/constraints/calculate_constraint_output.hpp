#pragma once

#include <Kokkos_Core.hpp>

#include "constraint_type.hpp"

#include "src/math/quaternion_operations.hpp"
#include "src/math/vector_operations.hpp"
#include "src/types.hpp"

namespace openturbine {

struct CalculateConstraintOutput {
    Kokkos::View<ConstraintType*>::const_type type;
    Kokkos::View<size_t*>::const_type target_node_index;
    Kokkos::View<double* [3][3]>::const_type axes;
    Kokkos::View<double* [7]>::const_type node_x0;     // Initial position
    Kokkos::View<double* [7]>::const_type node_u;      // Displacement
    Kokkos::View<double* [6]>::const_type node_udot;   // Velocity
    Kokkos::View<double* [6]>::const_type node_uddot;  // Acceleration
    Kokkos::View<double* [3]> outputs;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        switch (type(i_constraint)) {
            case ConstraintType::kRevoluteJoint: {
                // Axis of rotation unit vector
                const auto joint_axis0_data = Kokkos::Array<double, 3>{
                    axes(i_constraint, 0, 0), axes(i_constraint, 0, 1), axes(i_constraint, 0, 2)
                };
                const auto joint_axis0 = View_3::const_type{joint_axis0_data.data()};

                // Target node index
                auto i_node = target_node_index(i_constraint);

                // Target node initial rotation
                const auto R0_data = Kokkos::Array<double, 4>{
                    node_u(i_node, 3), node_u(i_node, 4), node_u(i_node, 5), node_u(i_node, 6)
                };
                const auto R0 = View_Quaternion::const_type{R0_data.data()};

                // Target node rotational displacement
                const auto R_data = Kokkos::Array<double, 4>{
                    node_u(i_node, 3), node_u(i_node, 4), node_u(i_node, 5), node_u(i_node, 6)
                };
                const auto R = View_Quaternion::const_type{R_data.data()};

                // Calculate current orientation
                auto RR0_data = Kokkos::Array<double, 4>{};
                auto RR0 = View_Quaternion(RR0_data.data());
                QuaternionCompose(R, R0, RR0);

                // Calculate rotational displacement as rotation vector
                auto RotVec_data = Kokkos::Array<double, 3>{};
                auto RotVec = View_3(RotVec_data.data());
                QuaternionToRotationVector(R, RotVec);

                // Target node rotational velocity vector
                auto omega_data = Kokkos::Array<double, 3>{
                    node_udot(i_node, 3), node_udot(i_node, 4), node_udot(i_node, 5)
                };
                auto omega = View_3{omega_data.data()};

                // Target node rotational acceleration vector
                auto omega_dot_data = Kokkos::Array<double, 3>{
                    node_uddot(i_node, 3), node_uddot(i_node, 4), node_uddot(i_node, 5)
                };
                auto omega_dot = View_3{omega_dot_data.data()};

                // Calculate joint axis in current configuration
                auto joint_axis_data = Kokkos::Array<double, 3>{};
                auto joint_axis = View_3{joint_axis_data.data()};
                RotateVectorByQuaternion(R, joint_axis0, joint_axis);

                // Calculate rotation about shaft axis
                auto angular_rotation = DotProduct(joint_axis, RotVec);

                // Angular velocity about joint axis (rad/s)
                auto angular_velocity = DotProduct(joint_axis, omega);

                // Angular acceleration about joint axis (rad/s)
                auto angular_acceleration = DotProduct(joint_axis, omega_dot);

                // Save outputs
                outputs(i_constraint, 0) = angular_rotation;
                outputs(i_constraint, 1) = angular_velocity;
                outputs(i_constraint, 2) = angular_acceleration;

            } break;
            default: {
                // Do nothing
            } break;
        }
    }
};

}  // namespace openturbine
