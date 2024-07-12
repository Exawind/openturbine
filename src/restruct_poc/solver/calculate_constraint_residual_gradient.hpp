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
    View_N Phi_;
    View_NxN B_;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        auto& cd = data(i_constraint);
        auto i_node1 = cd.base_node_index;
        auto i_node2 = cd.target_node_index;

        // Initial difference between nodes
        auto X0_data = Kokkos::Array<double, 3>{cd.X0[0], cd.X0[1], cd.X0[2]};
        auto X0 = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{X0_data.data()};

        // Base node displacement
        auto u1_data = Kokkos::Array<double, 3>{};
        auto R1_data = Kokkos::Array<double, 4>{};
        switch (cd.type) {
            case ConstraintType::kFixedBC: {
                u1_data = Kokkos::Array<double, 3>{0., 0., 0.};
                R1_data = Kokkos::Array<double, 4>{1., 0., 0., 0.};
            } break;
            case ConstraintType::kPrescribedBC: {
                u1_data = Kokkos::Array<double, 3>{
                    constraint_u(i_constraint, 0), constraint_u(i_constraint, 1),
                    constraint_u(i_constraint, 2)};
                R1_data = Kokkos::Array<double, 4>{
                    constraint_u(i_constraint, 3), constraint_u(i_constraint, 4),
                    constraint_u(i_constraint, 5), constraint_u(i_constraint, 6)};
            } break;
            default: {
                u1_data = Kokkos::Array<double, 3>{
                    node_u(i_node1, 0), node_u(i_node1, 1), node_u(i_node1, 2)};
                R1_data = Kokkos::Array<double, 4>{
                    node_u(i_node1, 3), node_u(i_node1, 4), node_u(i_node1, 5), node_u(i_node1, 6)};
            }
        }
        auto u1 = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{u1_data.data()};
        auto R1 = Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{R1_data.data()};

        // Target node displacement
        auto R2_data = Kokkos::Array<double, 4>{
            node_u(i_node2, 3), node_u(i_node2, 4), node_u(i_node2, 5), node_u(i_node2, 6)};
        auto R2 = Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{R2_data.data()};
        auto u2_data =
            Kokkos::Array<double, 3>{node_u(i_node2, 0), node_u(i_node2, 1), node_u(i_node2, 2)};
        auto u2 = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{u2_data.data()};

        // Rotation control
        auto RC_data = Kokkos::Array<double, 4>{1., 0., 0., 0};
        auto RC = Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{RC_data.data()};
        auto RCt_data = Kokkos::Array<double, 4>{1., 0., 0., 0.};
        auto RCt = Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{RCt_data.data()};
        auto RV_data = Kokkos::Array<double, 3>{};
        auto RV = Kokkos::View<double[4], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{RV_data.data()};

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

        auto V3_data = Kokkos::Array<double, 3>{};
        auto V3 = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{V3_data.data()};

        // Cylindrical constraint data
        auto x0_data = Kokkos::Array<double, 3>{cd.axis_x[0], cd.axis_x[1], cd.axis_x[2]};
        auto y0_data = Kokkos::Array<double, 3>{cd.axis_y[0], cd.axis_y[1], cd.axis_y[2]};
        auto z0_data = Kokkos::Array<double, 3>{cd.axis_z[0], cd.axis_z[1], cd.axis_z[2]};
        auto x0 = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{x0_data.data()};
        auto y0 = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{y0_data.data()};
        auto z0 = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{z0_data.data()};
        auto x_data = Kokkos::Array<double, 3>{};
        auto y_data = Kokkos::Array<double, 3>{};
        auto z_data = Kokkos::Array<double, 3>{};
        auto x = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{x_data.data()};
        auto y = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{y_data.data()};
        auto z = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{z_data.data()};
        auto xcy_data = Kokkos::Array<double, 3>{};
        auto xcz_data = Kokkos::Array<double, 3>{};
        auto xcy = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{xcy_data.data()};
        auto xcz = Kokkos::View<double[3], Kokkos::MemoryTraits<Kokkos::Unmanaged>>{xcz_data.data()};

        //----------------------------------------------------------------------
        // Residual Vector
        //----------------------------------------------------------------------

        // Extract residual rows relevant to this constraint
        auto Phi = Kokkos::subview(Phi_, cd.row_range);

        // Position residual = u2 + X0 - u1 - R1*X0
        QuaternionInverse(R1, R1t);
        RotateVectorByQuaternion(R1, X0, R1_X0);
        for (int i = 0; i < 3; ++i) {
            Phi(i) = u2(i) + X0(i) - u1(i) - R1_X0(i);
        }

        // Angular residual
        if (cd.type == ConstraintType::kCylindrical) {
            RotateVectorByQuaternion(R1, x0, x);
            RotateVectorByQuaternion(R2, y0, y);
            RotateVectorByQuaternion(R2, z0, z);
            Phi(3) = DotProduct(z, x);  // dot(R2 * z0_hat, R1 * x0_hat)
            Phi(4) = DotProduct(y, x);  // dot(R2 * y0_hat, R1 * x0_hat)
        } else {
            // If this is a rotation control constraint, calculate RC from control and axis
            if (cd.type == ConstraintType::kRotationControl) {
                RV(0) = cd.axis_x[0] * control(i_constraint);
                RV(1) = cd.axis_x[1] * control(i_constraint);
                RV(2) = cd.axis_x[2] * control(i_constraint);
                RotationVectorToQuaternion(RV, RC);
                QuaternionInverse(RC, RCt);
            }

            // axial(R2*inv(RC)*inv(R1))
            QuaternionCompose(R2, RCt, R2_RCt);
            QuaternionCompose(R2_RCt, R1t, R2_RCt_R1t);
            QuaternionToRotationMatrix(R2_RCt_R1t, C);
            AxialVectorOfMatrix(C, V3);
            for (int i = 0; i < 3; ++i) {
                Phi(i + 3) = V3(i);
            }
        }

        //----------------------------------------------------------------------
        // Constraint Gradient Matrix
        //----------------------------------------------------------------------

        //---------------------------------
        // Target Node
        //---------------------------------

        // Extract gradient block for target node of this constraint
        auto i_col = i_node2 * kLieAlgebraComponents;
        auto B = Kokkos::subview(
            B_, cd.row_range, Kokkos::make_pair(i_col, i_col + kLieAlgebraComponents)
        );

        // B11 = I
        for (int i = 0; i < 3; ++i) {
            B(i, i) = 1.;
        }

        if (cd.type == ConstraintType::kCylindrical) {
            // B22 = -cross(R1 * x0_hat, transpose(R2 * z0_hat))
            // B22 = -cross(R1 * x0_hat, transpose(R2 * y0_hat))
            CrossProduct(x, z, xcz);
            CrossProduct(x, y, xcy);
            for (int j = 0; j < 3; ++j) {
                B(3, j + 3) = -xcz(j);
                B(4, j + 3) = -xcy(j);
            }

        } else {
            // B22 = AX(R1*RC*inv(R2)) = transpose(AX(R2*inv(RC)*inv(R1)))
            AX_Matrix(C, A);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    B(i + 3, j + 3) = A(j, i);
                }
            }
        }

        //---------------------------------
        // Base Node
        //---------------------------------

        switch (cd.type) {
            case ConstraintType::kFixedBC:
            case ConstraintType::kPrescribedBC:
                return;
            default:
                break;
        }

        // Extract gradient block for base node of this constraint
        i_col = i_node1 * kLieAlgebraComponents;
        B = Kokkos::subview(
            B_, cd.row_range, Kokkos::make_pair(i_col, i_col + kLieAlgebraComponents)
        );

        // B11 = -I
        for (int i = 0; i < 3; ++i) {
            B(i, i) = -1.;
        }

        if (cd.type == ConstraintType::kCylindrical) {
            // B22 = cross(R1 * x0_hat, transpose(R2 * z0_hat))
            // B22 = cross(R1 * x0_hat, transpose(R2 * y0_hat))
            for (int j = 0; j < 3; ++j) {
                B(3, j + 3) = xcz(j);
                B(4, j + 3) = xcy(j);
            }

        } else {
            // B12 = tilde(R1*X0)
            VecTilde(R1_X0, A);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    B(i, j + 3) = A(i, j);
                }
            }

            // B22 = -AX(R2*inv(RC)*inv(R1))
            AX_Matrix(C, A);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    B(i + 3, j + 3) = -A(i, j);
                }
            }
        }
    }
};

}  // namespace openturbine
