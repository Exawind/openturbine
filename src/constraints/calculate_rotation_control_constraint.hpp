#pragma once

#include <KokkosBatched_Copy_Decl.hpp>
#include <Kokkos_Core.hpp>

#include "math/matrix_operations.hpp"
#include "math/quaternion_operations.hpp"
#include "math/vector_operations.hpp"

namespace openturbine {

template <typename DeviceType>
KOKKOS_INLINE_FUNCTION void CalculateRotationControlConstraint(
    const typename Kokkos::View<double[3], DeviceType>::const_type& X0,
    const typename Kokkos::View<double[3][3], DeviceType>::const_type& axes,
    const typename Kokkos::View<double[7], DeviceType>::const_type& inputs,
    const typename Kokkos::View<double[7], DeviceType>::const_type& base_node_u,
    const typename Kokkos::View<double[7], DeviceType>::const_type& target_node_u,
    const Kokkos::View<double[6], DeviceType>& residual_terms,
    const Kokkos::View<double[6][6], DeviceType>& base_gradient_terms,
    const Kokkos::View<double[6][6], DeviceType>& target_gradient_terms
) {
    const auto ub_data = Kokkos::Array<double, 3>{base_node_u(0), base_node_u(1), base_node_u(2)};
    const auto Rb_data =
        Kokkos::Array<double, 4>{base_node_u(3), base_node_u(4), base_node_u(5), base_node_u(6)};
    const auto ut_data =
        Kokkos::Array<double, 3>{target_node_u(0), target_node_u(1), target_node_u(2)};
    const auto Rt_data = Kokkos::Array<double, 4>{
        target_node_u(3), target_node_u(4), target_node_u(5), target_node_u(6)
    };
    auto AX_data = Kokkos::Array<double, 3>{};
    auto RV_data = Kokkos::Array<double, 3>{};
    auto Rc_data = Kokkos::Array<double, 4>{};
    auto RcT_data = Kokkos::Array<double, 4>{};
    auto RbT_data = Kokkos::Array<double, 4>{};
    auto Rb_X0_data = Kokkos::Array<double, 4>{};
    auto Rt_RcT_data = Kokkos::Array<double, 4>{};
    auto Rt_RcT_RbT_data = Kokkos::Array<double, 4>{};
    auto A_data = Kokkos::Array<double, 9>{};
    auto C_data = Kokkos::Array<double, 9>{};
    auto V3_data = Kokkos::Array<double, 3>{};

    const auto ub = typename Kokkos::View<double[3], DeviceType>::const_type{ub_data.data()};
    const auto Rb = typename Kokkos::View<double[4], DeviceType>::const_type{Rb_data.data()};
    const auto ut = typename Kokkos::View<double[3], DeviceType>::const_type{ut_data.data()};
    const auto Rt = typename Kokkos::View<double[4], DeviceType>::const_type{Rt_data.data()};
    const auto AX = Kokkos::View<double[3], DeviceType>{AX_data.data()};
    const auto RV = Kokkos::View<double[3], DeviceType>{RV_data.data()};
    const auto Rc = Kokkos::View<double[4], DeviceType>{Rc_data.data()};
    const auto RcT = Kokkos::View<double[4], DeviceType>{RcT_data.data()};
    const auto RbT = Kokkos::View<double[4], DeviceType>{RbT_data.data()};
    const auto Rb_X0 = Kokkos::View<double[4], DeviceType>{Rb_X0_data.data()};
    const auto Rt_RcT = Kokkos::View<double[4], DeviceType>{Rt_RcT_data.data()};
    const auto Rt_RcT_RbT = Kokkos::View<double[4], DeviceType>{Rt_RcT_RbT_data.data()};
    const auto A = Kokkos::View<double[3][3], DeviceType>{A_data.data()};
    const auto C = Kokkos::View<double[3][3], DeviceType>{C_data.data()};
    const auto V3 = Kokkos::View<double[3], DeviceType>{V3_data.data()};

    //----------------------------------------------------------------------
    // Position residual
    //----------------------------------------------------------------------

    // Phi(0:3) = ut + X0 - ub - Rb*X0
    RotateVectorByQuaternion(Rb, X0, Rb_X0);
    for (auto i = 0; i < 3; ++i) {
        residual_terms(i) = ut(i) + X0(i) - ub(i) - Rb_X0(i);
    }

    //----------------------------------------------------------------------
    // Rotation residual
    //----------------------------------------------------------------------

    auto rotation_command = inputs(0);

    // Copy rotation axis for this constraint
    for (auto i = 0U; i < 3U; ++i) {
        AX(i) = axes(0, i);
        RV(i) = AX(i) * rotation_command;
    }

    // Convert scaled axis to quaternion and calculate inverse
    RotationVectorToQuaternion(RV, Rc);
    QuaternionInverse(Rc, RcT);

    // Phi(3:6) = axial(Rt*inv(Rc)*inv(Rb))
    QuaternionInverse(Rb, RbT);
    QuaternionCompose(Rt, RcT, Rt_RcT);
    QuaternionCompose(Rt_RcT, RbT, Rt_RcT_RbT);
    QuaternionToRotationMatrix(Rt_RcT_RbT, C);
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

    // B(3:6,3:6) = AX(Rb*Rc*inv(Rt)) = transpose(AX(Rt*inv(Rc)*inv(Rb)))
    AX_Matrix(C, A);
    KokkosBatched::SerialCopy<KokkosBatched::Trans::Transpose>::invoke(
        A, Kokkos::subview(target_gradient_terms, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6))
    );

    //---------------------------------
    // Base Node
    //---------------------------------
    // B(0:3,0:3) = -I
    for (auto component = 0; component < 3; ++component) {
        base_gradient_terms(component, component) = -1.;
    }

    // B(0:3,3:6) = tilde(Rb*X0)
    VecTilde(Rb_X0, A);
    KokkosBatched::SerialCopy<>::invoke(
        A, Kokkos::subview(base_gradient_terms, Kokkos::make_pair(0, 3), Kokkos::make_pair(3, 6))
    );

    // B(3:6,3:6) = -AX(Rt*inv(Rc)*inv(Rb))
    AX_Matrix(C, A);
    for (auto component_1 = 0; component_1 < 3; ++component_1) {
        for (auto component_2 = 0; component_2 < 3; ++component_2) {
            base_gradient_terms(component_1 + 3, component_2 + 3) = -A(component_1, component_2);
        }
    }
}
}  // namespace openturbine
