#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"
#include "math/matrix_operations.hpp"
#include "math/quaternion_operations.hpp"
#include "math/vector_operations.hpp"

namespace openturbine {

KOKKOS_FUNCTION
inline void CalculatePrescribedBC3DOFConstraint(
    const Kokkos::View<double[3]>::const_type& X0, const Kokkos::View<double[7]>::const_type& inputs,
    const Kokkos::View<double[7]>::const_type& node_u, const Kokkos::View<double[6]>& residual_terms,
    const Kokkos::View<double[6][6]>& target_gradient_terms
) {
    const auto u1_data = Kokkos::Array<double, 3>{inputs(0), inputs(1), inputs(2)};
    const auto R1_data = Kokkos::Array<double, 4>{inputs(3), inputs(4), inputs(5), inputs(6)};
    const auto u2_data = Kokkos::Array<double, 3>{node_u(0), node_u(1), node_u(2)};
    const auto R2_data = Kokkos::Array<double, 4>{node_u(3), node_u(4), node_u(5), node_u(6)};
    auto R1t_data = Kokkos::Array<double, 4>{};
    auto R1_X0_data = Kokkos::Array<double, 4>{};
    auto R2_R1t_data = Kokkos::Array<double, 4>{};
    auto A_data = Kokkos::Array<double, 9>{};
    auto C_data = Kokkos::Array<double, 9>{};
    auto V3_data = Kokkos::Array<double, 3>{};

    const auto u1 = Kokkos::View<double[3]>::const_type{u1_data.data()};
    const auto R1 = Kokkos::View<double[4]>::const_type{R1_data.data()};
    const auto u2 = Kokkos::View<double[3]>::const_type{u2_data.data()};
    const auto R2 = Kokkos::View<double[4]>::const_type{R2_data.data()};
    const auto R1t = Kokkos::View<double[4]>{R1t_data.data()};
    const auto R1_X0 = Kokkos::View<double[4]>{R1_X0_data.data()};
    const auto R2_R1t = Kokkos::View<double[4]>{R2_R1t_data.data()};
    const auto A = Kokkos::View<double[3][3]>{A_data.data()};
    const auto C = Kokkos::View<double[3][3]>{C_data.data()};
    const auto V3 = Kokkos::View<double[3]>{V3_data.data()};

    //----------------------------------------------------------------------
    // Residual Vector
    //----------------------------------------------------------------------

    // Phi(0:3) = u2 + X0 - u1 - R1*X0
    QuaternionInverse(R1, R1t);
    RotateVectorByQuaternion(R1, X0, R1_X0);
    for (int i = 0; i < 3; ++i) {
        residual_terms(i) = u2(i) + X0(i) - u1(i) - R1_X0(i);
    }

    //----------------------------------------------------------------------
    // Constraint Gradient Matrix
    //----------------------------------------------------------------------

    //---------------------------------
    // Target Node
    //---------------------------------

    // B(0:3,0:3) = I
    for (int i = 0; i < 3; ++i) {
        target_gradient_terms(i, i) = 1.;
    }
}

}  // namespace openturbine
