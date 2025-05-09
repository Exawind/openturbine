#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"
#include "math/quaternion_operations.hpp"
#include "math/vector_operations.hpp"

namespace openturbine {

template <typename DeviceType>
KOKKOS_INLINE_FUNCTION
void CalculateRevoluteJointConstraint(
    const typename Kokkos::View<double[3], DeviceType>::const_type& X0,
    const typename Kokkos::View<double[3][3], DeviceType>::const_type& axes,
    const typename Kokkos::View<double[7], DeviceType>::const_type& base_node_u,
    const typename Kokkos::View<double[7], DeviceType>::const_type& target_node_u,
    const Kokkos::View<double[6], DeviceType>& residual_terms,
    const Kokkos::View<double[6][6], DeviceType>& base_gradient_terms,
    const Kokkos::View<double[6][6], DeviceType>& target_gradient_terms
) {
    const auto u1_data = Kokkos::Array<double, 3>{base_node_u(0), base_node_u(1), base_node_u(2)};
    const auto R1_data =
        Kokkos::Array<double, 4>{base_node_u(3), base_node_u(4), base_node_u(5), base_node_u(6)};
    const auto u2_data =
        Kokkos::Array<double, 3>{target_node_u(0), target_node_u(1), target_node_u(2)};
    const auto R2_data = Kokkos::Array<double, 4>{
        target_node_u(3), target_node_u(4), target_node_u(5), target_node_u(6)
    };
    const auto x0_data = Kokkos::Array<double, 3>{axes(0, 0), axes(0, 1), axes(0, 2)};
    const auto y0_data = Kokkos::Array<double, 3>{axes(1, 0), axes(1, 1), axes(1, 2)};
    const auto z0_data = Kokkos::Array<double, 3>{axes(2, 0), axes(2, 1), axes(2, 2)};
    auto R1t_data = Kokkos::Array<double, 4>{};
    auto R1_X0_data = Kokkos::Array<double, 4>{};
    auto x_data = Kokkos::Array<double, 3>{};
    auto y_data = Kokkos::Array<double, 3>{};
    auto z_data = Kokkos::Array<double, 3>{};
    auto xcy_data = Kokkos::Array<double, 3>{};
    auto xcz_data = Kokkos::Array<double, 3>{};

    const auto u1 = typename Kokkos::View<double[3], DeviceType>::const_type{u1_data.data()};
    const auto R1 = typename Kokkos::View<double[4], DeviceType>::const_type{R1_data.data()};
    const auto u2 = typename Kokkos::View<double[3], DeviceType>::const_type{u2_data.data()};
    const auto R2 = typename Kokkos::View<double[4], DeviceType>::const_type{R2_data.data()};
    const auto x0 = typename Kokkos::View<double[3], DeviceType>::const_type{x0_data.data()};
    const auto y0 = typename Kokkos::View<double[3], DeviceType>::const_type{y0_data.data()};
    const auto z0 = typename Kokkos::View<double[3], DeviceType>::const_type{z0_data.data()};
    const auto R1t = Kokkos::View<double[4], DeviceType>{R1t_data.data()};
    const auto R1_X0 = Kokkos::View<double[4], DeviceType>{R1_X0_data.data()};
    const auto x = Kokkos::View<double[3], DeviceType>{x_data.data()};
    const auto y = Kokkos::View<double[3], DeviceType>{y_data.data()};
    const auto z = Kokkos::View<double[3], DeviceType>{z_data.data()};
    const auto xcy = Kokkos::View<double[3], DeviceType>{xcy_data.data()};
    const auto xcz = Kokkos::View<double[3], DeviceType>{xcz_data.data()};

    //----------------------------------------------------------------------
    // Residual Vector, Phi
    //----------------------------------------------------------------------

    // Phi(0:3) = u2 + X0 - u1 - R1*X0
    QuaternionInverse(R1, R1t);
    RotateVectorByQuaternion(R1, X0, R1_X0);
    for (int i = 0; i < 3; ++i) {
        residual_terms(i) = u2(i) + X0(i) - u1(i) - R1_X0(i);
    }

    // Angular residual
    RotateVectorByQuaternion(R1, x0, x);
    RotateVectorByQuaternion(R2, y0, y);
    RotateVectorByQuaternion(R2, z0, z);
    // Phi(3) = dot(R2 * z0_hat, R1 * x0_hat)
    residual_terms(3) = DotProduct(z, x);
    // Phi(4) = dot(R2 * y0_hat, R1 * x0_hat)
    residual_terms(4) = DotProduct(y, x);

    //----------------------------------------------------------------------
    // Constraint Gradient Matrix, B
    //----------------------------------------------------------------------

    //---------------------------------
    // Target Node
    //---------------------------------
    for (int i = 0; i < 3; ++i) {
        target_gradient_terms(i, i) = 1.;
    }

    // B(3, 3:6) = -cross(R1 * x0_hat, transpose(R2 * z0_hat))
    CrossProduct(x, z, xcz);
    // B(4, 3:6) = -cross(R1 * x0_hat, transpose(R2 * y0_hat))
    CrossProduct(x, y, xcy);
    for (int j = 0; j < 3; ++j) {
        target_gradient_terms(3, j + 3) = -xcz(j);
        target_gradient_terms(4, j + 3) = -xcy(j);
    }
    //---------------------------------
    // Base Node
    //---------------------------------
    // B(0:3,0:3) = -I
    for (int i = 0; i < 3; ++i) {
        base_gradient_terms(i, i) = -1.;
    }

    // B(3,3:6) = cross(R1 * x0_hat, transpose(R2 * z0_hat))
    // B(4,3:6) = cross(R1 * x0_hat, transpose(R2 * y0_hat))
    for (int j = 0; j < 3; ++j) {
        base_gradient_terms(3, j + 3) = xcz(j);
        base_gradient_terms(4, j + 3) = xcy(j);
    }
}
}  // namespace openturbine
