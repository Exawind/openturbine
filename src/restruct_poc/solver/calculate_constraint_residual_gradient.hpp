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
    Kokkos::View<double* [6][12]> gradient_terms;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        auto& cd = data(i_constraint);
        auto i_node1 = cd.base_node_index;
        auto i_node2 = cd.target_node_index;

        // Initial difference between nodes
        auto X0_data = Kokkos::Array<double, 3>{cd.X0[0], cd.X0[1], cd.X0[2]};
        auto X0 = View_3{X0_data.data()};

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
        auto u1 = View_3{u1_data.data()};
        auto R1 = Kokkos::View<double[4]>{R1_data.data()};

        // Target node displacement
        auto R2_data = Kokkos::Array<double, 4>{
            node_u(i_node2, 3), node_u(i_node2, 4), node_u(i_node2, 5), node_u(i_node2, 6)};
        auto R2 = Kokkos::View<double[4]>{R2_data.data()};
        auto u2_data =
            Kokkos::Array<double, 3>{node_u(i_node2, 0), node_u(i_node2, 1), node_u(i_node2, 2)};
        auto u2 = View_3{u2_data.data()};

        // Rotation control
        auto RC_data = Kokkos::Array<double, 4>{1., 0., 0., 0};
        auto RC = Kokkos::View<double[4]>{RC_data.data()};
        auto RCt_data = Kokkos::Array<double, 4>{1., 0., 0., 0.};
        auto RCt = Kokkos::View<double[4]>{RCt_data.data()};
        auto RV_data = Kokkos::Array<double, 3>{};
        auto RV = Kokkos::View<double[4]>{RV_data.data()};

        auto R1t_data = Kokkos::Array<double, 4>{};
        auto R1t = Kokkos::View<double[4]>{R1t_data.data()};

        auto R2t_data = Kokkos::Array<double, 4>{};
        auto R2t = Kokkos::View<double[4]>{R2t_data.data()};

        auto R1_X0_data = Kokkos::Array<double, 3>{};
        auto R1_X0 = Kokkos::View<double[4]>{R1_X0_data.data()};

        auto R2_R1t_data = Kokkos::Array<double, 4>{};
        auto R2_R1t = Kokkos::View<double[4]>{R2_R1t_data.data()};

        auto R2_RCt_data = Kokkos::Array<double, 4>{};
        auto R2_RCt = Kokkos::View<double[4]>{R2_RCt_data.data()};

        auto R1_R2t_data = Kokkos::Array<double, 4>{};
        auto R1_R2t = Kokkos::View<double[4]>{R1_R2t_data.data()};

        auto R2_RCt_R1t_data = Kokkos::Array<double, 4>{};
        auto R2_RCt_R1t = Kokkos::View<double[4]>{R2_RCt_R1t_data.data()};

        auto A_data = Kokkos::Array<double, 9>{};
        auto A = View_3x3{A_data.data()};

        auto C_data = Kokkos::Array<double, 9>{};
        auto C = View_3x3{C_data.data()};

        auto Ct_data = Kokkos::Array<double, 9>{};
        auto Ct = View_3x3{Ct_data.data()};

        auto V3_data = Kokkos::Array<double, 3>{};
        auto V3 = View_3{V3_data.data()};

        // Cylindrical constraint data
        auto x0_data = Kokkos::Array<double, 3>{cd.axis_x[0], cd.axis_x[1], cd.axis_x[2]};
        auto y0_data = Kokkos::Array<double, 3>{cd.axis_y[0], cd.axis_y[1], cd.axis_y[2]};
        auto z0_data = Kokkos::Array<double, 3>{cd.axis_z[0], cd.axis_z[1], cd.axis_z[2]};
        auto x0 = View_3{x0_data.data()};
        auto y0 = View_3{y0_data.data()};
        auto z0 = View_3{z0_data.data()};
        auto x_data = Kokkos::Array<double, 3>{};
        auto y_data = Kokkos::Array<double, 3>{};
        auto z_data = Kokkos::Array<double, 3>{};
        auto x = View_3{x_data.data()};
        auto y = View_3{y_data.data()};
        auto z = View_3{z_data.data()};
        auto xcy_data = Kokkos::Array<double, 3>{};
        auto xcz_data = Kokkos::Array<double, 3>{};
        auto xcy = View_3{xcy_data.data()};
        auto xcz = View_3{xcz_data.data()};

        //----------------------------------------------------------------------
        // Residual Vector
        //----------------------------------------------------------------------

        // Extract residual rows relevant to this constraint
        auto Phi = Kokkos::subview(Phi_, cd.row_range);

        // Phi(0:3) = u2 + X0 - u1 - R1*X0
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
            // Phi(3) = dot(R2 * z0_hat, R1 * x0_hat)
            Phi(3) = DotProduct(z, x);
            // Phi(4) = dot(R2 * y0_hat, R1 * x0_hat)
            Phi(4) = DotProduct(y, x);
        } else {
            // If this is a rotation control constraint, calculate RC from control and axis
            if (cd.type == ConstraintType::kRotationControl) {
                RV(0) = cd.axis_x[0] * control(i_constraint);
                RV(1) = cd.axis_x[1] * control(i_constraint);
                RV(2) = cd.axis_x[2] * control(i_constraint);
                RotationVectorToQuaternion(RV, RC);
                QuaternionInverse(RC, RCt);
            }

            // Phi(3:6) = axial(R2*inv(RC)*inv(R1))
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
        auto i_col = (i_node2 < i_node1 || i_node1 < 0) ? 0 : kLieAlgebraComponents;
        auto B = Kokkos::subview(
            gradient_terms, i_constraint, Kokkos::ALL,
            Kokkos::make_pair(i_col, i_col + kLieAlgebraComponents)
        );

        // B(0:3,0:3) = I
        for (int i = 0; i < 3; ++i) {
            B(i, i) = 1.;
        }

        if (cd.type == ConstraintType::kCylindrical) {
            // B(3,3:6) = -cross(R1 * x0_hat, transpose(R2 * z0_hat))
            CrossProduct(x, z, xcz);
            // B(4,3:6) = -cross(R1 * x0_hat, transpose(R2 * y0_hat))
            CrossProduct(x, y, xcy);
            for (int j = 0; j < 3; ++j) {
                B(3, j + 3) = -xcz(j);
                B(4, j + 3) = -xcy(j);
            }

        } else {
            // B(3:6,3:6) = AX(R1*RC*inv(R2)) = transpose(AX(R2*inv(RC)*inv(R1)))
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
        i_col = (i_node1 < i_node2) ? 0 : kLieAlgebraComponents;
        B = Kokkos::subview(
            gradient_terms, i_constraint, Kokkos::ALL,
            Kokkos::make_pair(i_col, i_col + kLieAlgebraComponents)
        );

        // B(0:3,0:3) = -I
        for (int i = 0; i < 3; ++i) {
            B(i, i) = -1.;
        }

        if (cd.type == ConstraintType::kCylindrical) {
            // B(3,3:6) = cross(R1 * x0_hat, transpose(R2 * z0_hat))
            // B(4,3:6) = cross(R1 * x0_hat, transpose(R2 * y0_hat))
            for (int j = 0; j < 3; ++j) {
                B(3, j + 3) = xcz(j);
                B(4, j + 3) = xcy(j);
            }

        } else {
            // B(0:3,3:6) = tilde(R1*X0)
            VecTilde(R1_X0, A);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    B(i, j + 3) = A(i, j);
                }
            }

            // B(3:6,3:6) = -AX(R2*inv(RC)*inv(R1))
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
