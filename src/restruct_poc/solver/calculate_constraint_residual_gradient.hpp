#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"

#include "src/restruct_poc/math/quaternion_operations.hpp"
#include "src/restruct_poc/math/vector_operations.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct CalculateConstraintResidualGradient {
    Kokkos::View<Constraints::Data*>::const_type data;
    View_N::const_type control;
    View_Nx7::const_type constraint_u;
    View_Nx7::const_type node_u;
    View_N Phi;
    View_NxN B;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        auto& cd = data(i_constraint);
        auto i_node1 = cd.base_node_index;
        auto i_node2 = cd.target_node_index;
        auto i_row = i_constraint * kLieAlgebraComponents;

        // Initial difference between nodes
        auto x0_data = Kokkos::Array<double, 3>{cd.X0[0], cd.X0[1], cd.X0[2]};
        auto X0 = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{x0_data.data()};

        // Base node displacement
        auto r1_data = Kokkos::Array<double, 4>{};
        auto u1_data = Kokkos::Array<double, 3>{};
        switch (cd.type) {
            case ConstraintType::FixedBC:
                u1_data = Kokkos::Array<double, 3>{0., 0., 0.};
                r1_data = Kokkos::Array<double, 4>{1., 0., 0., 0.};
                break;
            case ConstraintType::PrescribedBC:
                u1_data = Kokkos::Array<double, 3>{
                    constraint_u(i_constraint, 0), constraint_u(i_constraint, 1),
                    constraint_u(i_constraint, 2)};
                r1_data = Kokkos::Array<double, 4>{
                    constraint_u(i_constraint, 3), constraint_u(i_constraint, 4),
                    constraint_u(i_constraint, 5), constraint_u(i_constraint, 6)};
                break;
            default:
                u1_data = Kokkos::Array<double, 3>{
                    node_u(i_node1, 0), node_u(i_node1, 1), node_u(i_node1, 2)};
                r1_data = Kokkos::Array<double, 4>{
                    node_u(i_node1, 3), node_u(i_node1, 4), node_u(i_node1, 5), node_u(i_node1, 6)};
        }
        auto u1 = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{u1_data.data()};
        auto R1 = Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{r1_data.data()};

        // Target node displacement
        auto r2_data = Kokkos::Array<double, 4>{
            node_u(i_node2, 3), node_u(i_node2, 4), node_u(i_node2, 5), node_u(i_node2, 6)};
        auto R2 = Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{r2_data.data()};
        auto u2_data =
            Kokkos::Array<double, 3>{node_u(i_node2, 0), node_u(i_node2, 1), node_u(i_node2, 2)};
        auto u2 = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{u2_data.data()};

        auto Phi_x_data = Kokkos::Array<double, 3>{};
        auto Phi_x =
            Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{Phi_x_data.data()};

        auto Phi_p_data = Kokkos::Array<double, 3>{};
        auto Phi_p =
            Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{Phi_p_data.data()};

        auto RV_data = Kokkos::Array<double, 3>{};
        auto RV = Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{RV_data.data()};

        auto RC_data = Kokkos::Array<double, 4>{};
        auto RC = Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{RC_data.data()};

        auto R1inv_data = Kokkos::Array<double, 4>{};
        auto R1inv =
            Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{R1inv_data.data()};

        auto R2inv_data = Kokkos::Array<double, 4>{};
        auto R2inv =
            Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{R2inv_data.data()};

        auto RCinv_data = Kokkos::Array<double, 4>{};
        auto RCinv =
            Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{RCinv_data.data()};

        auto R1_X0_data = Kokkos::Array<double, 3>{};
        auto R1_X0 =
            Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{R1_X0_data.data()};

        auto R2_R1inv_data = Kokkos::Array<double, 4>{};
        auto R2_R1inv =
            Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{R2_R1inv_data.data()};

        auto R2_RCinv_data = Kokkos::Array<double, 4>{};
        auto R2_RCinv =
            Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{R2_RCinv_data.data()};

        auto R1_R2inv_data = Kokkos::Array<double, 4>{};
        auto R1_R2inv =
            Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{R1_R2inv_data.data()};

        auto R2_RCinv_R1inv_data = Kokkos::Array<double, 4>{};
        auto R2_RCinv_R1inv = Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{
            R2_RCinv_R1inv_data.data()};

        auto A_data = Kokkos::Array<double, 9>{};
        auto A = Kokkos::View<double[3][3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{A_data.data()};

        auto C_data = Kokkos::Array<double, 9>{};
        auto C = Kokkos::View<double[3][3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{C_data.data()};

        //----------------------------------------------------------------------
        // Residual Vector
        //----------------------------------------------------------------------

        // Phi_x = u2 + X0 - u1 - R1*X0
        RotateVectorByQuaternion(R1, X0, R1_X0);
        for (int i = 0; i < 3; ++i) {
            Phi_x(i) = u2(i) + X0(i) - u1(i) - R1_X0(i);
        }

        QuaternionInverse(R1, R1inv);
        switch (cd.type) {
            case ConstraintType::RotationControl:
                // Phi_p = Axial(R2*inv(RC)*inv(R1))
                RV(0) = cd.axis[0] * control(i_constraint);
                RV(1) = cd.axis[1] * control(i_constraint);
                RV(2) = cd.axis[2] * control(i_constraint);
                RotationVectorToQuaternion(RV, RC);
                QuaternionInverse(RC, RCinv);
                QuaternionCompose(R2, RCinv, R2_RCinv);
                QuaternionCompose(R2_RCinv, R1inv, R2_RCinv_R1inv);
                QuaternionToRotationMatrix(R2_RCinv_R1inv, C);
                AxialVectorOfMatrix(C, Phi_p);
            case ConstraintType::Cylindrical:
                // Phi_p = Axial(R2*inv(RC)*inv(R1))
            default:
                // Phi_p = Axial(R2*inv(R1))
                QuaternionCompose(R2, R1inv, R2_R1inv);
                QuaternionToRotationMatrix(R2_R1inv, C);
                AxialVectorOfMatrix(C, Phi_p);
        }

        Phi(i_row + 0) += Phi_x(0);
        Phi(i_row + 1) += Phi_x(1);
        Phi(i_row + 2) += Phi_x(2);
        Phi(i_row + 3) += Phi_p(0);
        Phi(i_row + 4) += Phi_p(1);
        Phi(i_row + 5) += Phi_p(2);

        //----------------------------------------------------------------------
        // Constraint Gradient Matrix
        //----------------------------------------------------------------------

        //---------------------------------
        // Target Node
        //---------------------------------

        // A = 1/2*(tr(R1*inv(R2))*I - R1*inv(R2))
        QuaternionInverse(R2, R2inv);
        QuaternionCompose(R1, R2inv, R1_R2inv);
        QuaternionToRotationMatrix(R1_R2inv, A);
        auto trace = A(0, 0) + A(1, 1) + A(2, 2);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                A(i, j) *= -1. / 2.;
            }
            A(i, i) += trace / 2.;
        }

        // Starting column index in B matrix
        auto i_col = i_node2 * kLieAlgebraComponents;

        // B11 = I
        for (int i = 0; i < 3; ++i) {
            B(i_row + i, i_col + i) = 1.;
        }

        // B22 = A
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                B(i_row + i + 3, i_col + j + 3) = A(i, j);
            }
        }

        //---------------------------------
        // Base Node
        //---------------------------------

        // If fixed or prescribed boundary, skip based node
        if (cd.type == ConstraintType::FixedBC || cd.type == ConstraintType::PrescribedBC)
            return;

        // Starting column index in B matrix
        i_col = i_node1 * kLieAlgebraComponents;

        // C = 1/2*(tr(R2*inv(R1))*I - R2*inv(R1))
        trace = C(0, 0) + C(1, 1) + C(2, 2);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                C(i, j) *= -1. / 2.;
            }
            C(i, i) += trace / 2.;
        }

        // B11 = -I
        for (int i = 0; i < 3; ++i) {
            B(i_row + i, i_col + i) = -1.;
        }

        // B12 = tilde(R1*X0)
        VecTilde(
            R1_X0,
            Kokkos::subview(B, Kokkos::pair(i_row, i_row + 3), Kokkos::pair(i_col + 3, i_col + 6))
        );

        // B22 = C
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                B(i_row + i + 3, i_col + j + 3) = -C(i, j);
            }
        }
    }
};

}  // namespace openturbine
