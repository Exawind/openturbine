#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"
#include "math/quaternion_operations.hpp"
#include "math/vector_operations.hpp"
#include "types.hpp"

namespace openturbine {

KOKKOS_FUNCTION
inline void CalculateRevoluteJointForce(
    const Kokkos::View<double[3][3]>::const_type& axes,
    const Kokkos::View<double[7]>::const_type& inputs,
    const Kokkos::View<double[7]>::const_type& node_u,
    const Kokkos::View<double[6]>& system_residual_terms
) {
    // Initial difference between nodes
    const auto X0_data = Kokkos::Array<double, 3>{axes(0, 0), axes(0, 1), axes(0, 2)};
    const auto R2_data = Kokkos::Array<double, 4>{node_u(3), node_u(4), node_u(5), node_u(6)};
    auto R2_X0_data = Kokkos::Array<double, 3>{};

    const auto X0 = View_3::const_type{X0_data.data()};
    const auto R2 = Kokkos::View<double[4]>::const_type{R2_data.data()};
    const auto R2_X0 = Kokkos::View<double[3]>{R2_X0_data.data()};

    //----------------------------------------------------------------------
    // Residual Vector
    //----------------------------------------------------------------------

    // Extract residual rows relevant to this constraint
    RotateVectorByQuaternion(R2, X0, R2_X0);

    // Take axis_x and rotate it to right orientation
    system_residual_terms(3) = R2_X0(0) * inputs(0);
    system_residual_terms(4) = R2_X0(1) * inputs(0);
    system_residual_terms(5) = R2_X0(2) * inputs(0);
}
}  // namespace openturbine
