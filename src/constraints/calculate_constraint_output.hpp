#pragma once

#include <Kokkos_Core.hpp>

#include "calculate_revolute_joint_output.hpp"
#include "constraint_type.hpp"

namespace openturbine {

template <typename DeviceType>
struct CalculateConstraintOutput {
    typename Kokkos::View<ConstraintType*, DeviceType>::const_type type;
    typename Kokkos::View<size_t*, DeviceType>::const_type target_node_index;
    typename Kokkos::View<double* [3][3], DeviceType>::const_type axes;
    typename Kokkos::View<double* [7], DeviceType>::const_type node_x0;     // Initial position
    typename Kokkos::View<double* [7], DeviceType>::const_type node_u;      // Displacement
    typename Kokkos::View<double* [6], DeviceType>::const_type node_udot;   // Velocity
    typename Kokkos::View<double* [6], DeviceType>::const_type node_uddot;  // Acceleration
    Kokkos::View<double* [3], DeviceType> outputs;

    KOKKOS_FUNCTION
    void operator()(const int constraint) const {
        if (type(constraint) == ConstraintType::RevoluteJoint) {
            CalculateRevoluteJointOutput<DeviceType>{constraint, target_node_index,
                                                     axes,       node_x0,
                                                     node_u,     node_udot,
                                                     node_uddot, outputs}();
            return;
        }
    }
};

}  // namespace openturbine
