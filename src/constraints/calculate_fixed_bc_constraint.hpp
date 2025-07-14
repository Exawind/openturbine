#pragma once

#include <KokkosBatched_Copy_Decl.hpp>
#include <Kokkos_Core.hpp>

#include "math/matrix_operations.hpp"
#include "math/quaternion_operations.hpp"

namespace openturbine {

template <typename DeviceType>
KOKKOS_INLINE_FUNCTION void CalculateFixedBCConstraint(
    const typename Kokkos::View<double[3], DeviceType>::const_type& X0,
    const typename Kokkos::View<double[7], DeviceType>::const_type& node_u,
    const typename Kokkos::View<double[6], DeviceType>& residual_terms,
    const Kokkos::View<double[6][6], DeviceType>& target_gradient_terms
) {
    constexpr auto u1_data = Kokkos::Array<double, 3>{0., 0., 0.};
    constexpr auto R1_data = Kokkos::Array<double, 4>{1., 0., 0., 0.};
    const auto u2_data = Kokkos::Array<double, 3>{node_u(0), node_u(1), node_u(2)};
    const auto R2_data = Kokkos::Array<double, 4>{node_u(3), node_u(4), node_u(5), node_u(6)};
    auto R1t_data = Kokkos::Array<double, 4>{};
    auto R1_X0_data = Kokkos::Array<double, 4>{};
    auto R2_R1t_data = Kokkos::Array<double, 4>{};
    auto A_data = Kokkos::Array<double, 9>{};
    auto C_data = Kokkos::Array<double, 9>{};
    auto V3_data = Kokkos::Array<double, 3>{};

    const auto u1 = typename Kokkos::View<double[3], DeviceType>::const_type{u1_data.data()};
    const auto R1 = typename Kokkos::View<double[4], DeviceType>::const_type{R1_data.data()};
    const auto u2 = typename Kokkos::View<double[3], DeviceType>::const_type{u2_data.data()};
    const auto R2 = typename Kokkos::View<double[4], DeviceType>::const_type{R2_data.data()};
    const auto R1t = Kokkos::View<double[4], DeviceType>{R1t_data.data()};
    const auto R1_X0 = Kokkos::View<double[4], DeviceType>{R1_X0_data.data()};
    const auto R2_R1t = Kokkos::View<double[4], DeviceType>{R2_R1t_data.data()};
    const auto A = Kokkos::View<double[3][3], DeviceType>{A_data.data()};
    const auto C = Kokkos::View<double[3][3], DeviceType>{C_data.data()};
    const auto V3 = Kokkos::View<double[3], DeviceType>{V3_data.data()};

    //----------------------------------------------------------------------
    // Residual Vector
    //----------------------------------------------------------------------

    // Phi(0:3) = u2 + X0 - u1 - R1*X0
    QuaternionInverse(R1, R1t);
    RotateVectorByQuaternion(R1, X0, R1_X0);
    for (auto component = 0; component < 3; ++component) {
        residual_terms(component) = u2(component) + X0(component) - u1(component) - R1_X0(component);
    }

    // Angular residual
    // Phi(3:6) = axial(R2*inv(RC)*inv(R1))
    QuaternionCompose(R2, R1t, R2_R1t);
    QuaternionToRotationMatrix(R2_R1t, C);
    AxialVectorOfMatrix(C, V3);
    KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose, 1>::invoke(
        V3, Kokkos::subview(residual_terms, Kokkos::make_pair(3, 6))
    );

    //----------------------------------------------------------------------
    // Constraint Gradient Matrix
    //----------------------------------------------------------------------

    //---------------------------------
    // Target Node
    //---------------------------------

    // B(0:3,0:3) = I
    for (auto component = 0; component < 3; ++component) {
        target_gradient_terms(component, component) = 1.;
    }

    // B(3:6,3:6) = AX(R1*RC*inv(R2)) = transpose(AX(R2*inv(RC)*inv(R1)))
    AX_Matrix(C, A);
    KokkosBatched::SerialCopy<KokkosBatched::Trans::Transpose>::invoke(
        A, Kokkos::subview(target_gradient_terms, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6))
    );
}
}  // namespace openturbine
