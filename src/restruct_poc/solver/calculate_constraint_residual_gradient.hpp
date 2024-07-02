#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"

#include "src/restruct_poc/math/quaternion_operations.hpp"
#include "src/restruct_poc/math/vector_operations.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct CalculateConstraintResidualGradient {
    Kokkos::View<Constraints::DeviceData*>::const_type data;
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

        auto RC_data = Kokkos::Array<double, 4>{1., 0., 0., 0};
        auto RC = Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{RC_data.data()};

        auto RCt_data = Kokkos::Array<double, 4>{1., 0., 0., 0.};
        auto RCt = Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{RCt_data.data()};

        auto R1t_data = Kokkos::Array<double, 4>{};
        auto R1t = Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{R1t_data.data()};

        auto R2t_data = Kokkos::Array<double, 4>{};
        auto R2t = Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{R2t_data.data()};

        auto R1_X0_data = Kokkos::Array<double, 3>{};
        auto R1_X0 =
            Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{R1_X0_data.data()};

        auto R2_R1t_data = Kokkos::Array<double, 4>{};
        auto R2_R1t =
            Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{R2_R1t_data.data()};

        auto R2_RCt_data = Kokkos::Array<double, 4>{};
        auto R2_RCt =
            Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{R2_RCt_data.data()};

        auto R1_R2t_data = Kokkos::Array<double, 4>{};
        auto R1_R2t =
            Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{R1_R2t_data.data()};

        auto R2_RCt_R1t_data = Kokkos::Array<double, 4>{};
        auto R2_RCt_R1t =
            Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{R2_RCt_R1t_data.data()};

        auto A_data = Kokkos::Array<double, 9>{};
        auto A = Kokkos::View<double[3][3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{A_data.data()};

        auto C_data = Kokkos::Array<double, 9>{};
        auto C = Kokkos::View<double[3][3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{C_data.data()};

        auto Ct_data = Kokkos::Array<double, 9>{};
        auto Ct =
            Kokkos::View<double[3][3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{Ct_data.data()};

        auto V_axial_data = Kokkos::Array<double, 3>{};
        auto V_axial =
            Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{V_axial_data.data()};

        //----------------------------------------------------------------------
        // Residual Vector
        //----------------------------------------------------------------------

        // Phi_x = u2 + X0 - u1 - R1*X0
        RotateVectorByQuaternion(R1, X0, R1_X0);
        for (int i = 0; i < 3; ++i) {
            Phi_x(i) = u2(i) + X0(i) - u1(i) - R1_X0(i);
        }

        QuaternionInverse(R1, R1t);
        switch (cd.type) {
            case ConstraintType::Cylindrical:
                // Phi_p =
                break;
            case ConstraintType::RotationControl:
                // Phi_p = axial(R2*inv(RC)*inv(R1))
                RV(0) = cd.axis[0] * control(i_constraint);
                RV(1) = cd.axis[1] * control(i_constraint);
                RV(2) = cd.axis[2] * control(i_constraint);
                RotationVectorToQuaternion(RV, RC);
                QuaternionInverse(RC, RCt);
                QuaternionCompose(R2, RCt, R2_RCt);
                QuaternionCompose(R2_RCt, R1t, R2_RCt_R1t);
                QuaternionToRotationMatrix(R2_RCt_R1t, C);
                AxialVectorOfMatrix(C, Phi_p);
                break;
            default:
                // Phi_p = axial(R2*inv(R1))
                QuaternionCompose(R2, R1t, R2_R1t);
                QuaternionToRotationMatrix(R2_R1t, C);
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

        // Starting column index in B matrix
        auto i_col = i_node2 * kLieAlgebraComponents;

        // B11 = I
        for (int i = 0; i < 3; ++i) {
            B(i_row + i, i_col + i) = 1.;
        }

        // B22 = AX(R1*inv(R2)) = transpose(AX(R2*inv(R1)))
        AX_Matrix(C, A);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                B(i_row + i + 3, i_col + j + 3) = A(j, i);
            }
        }

        //---------------------------------
        // Base Node
        //---------------------------------

        switch (cd.type) {
            case ConstraintType::FixedBC:
            case ConstraintType::PrescribedBC:
                return;
            default:
                break;
        }

        // Starting column index in B matrix
        i_col = i_node1 * kLieAlgebraComponents;

        // B11 = -I
        for (int i = 0; i < 3; ++i) {
            B(i_row + i, i_col + i) = -1.;
        }

        // B12 = tilde(R1*X0)
        VecTilde(R1_X0, A);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                B(i_row + i, i_col + j + 3) = A(i, j);
            }
        }

        // B22 = -AX(R2*inv(R1))
        AX_Matrix(C, A);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                B(i_row + i + 3, i_col + j + 3) = -A(i, j);
            }
        }
    }
};

}  // namespace openturbine
