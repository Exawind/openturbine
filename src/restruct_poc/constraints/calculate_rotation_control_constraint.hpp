#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"

#include "src/restruct_poc/math/quaternion_operations.hpp"
#include "src/restruct_poc/math/vector_operations.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct CalculateRotationControlConstraint {
    Kokkos::View<size_t*>::const_type base_node_index;
    Kokkos::View<size_t*>::const_type target_node_index;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type row_range;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type base_node_col_range;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type target_node_col_range;
    Kokkos::View<double* [3]>::const_type X0_;
    Kokkos::View<double* [3][3]>::const_type axes;
    View_N::const_type control;
    View_Nx7::const_type constraint_u;
    View_Nx7::const_type node_u;
    View_N Phi_;
    Kokkos::View<double* [6][12]> gradient_terms;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        const auto i_node1 = base_node_index(i_constraint);
        const auto i_node2 = target_node_index(i_constraint);

        // Initial difference between nodes
        const auto X0_data = Kokkos::Array<double, 3>{
            X0_(i_constraint, 0), X0_(i_constraint, 1), X0_(i_constraint, 2)};
        const auto X0 = View_3::const_type{X0_data.data()};

        // Base node displacement
        const auto u1_data =
            Kokkos::Array<double, 3>{node_u(i_node1, 0), node_u(i_node1, 1), node_u(i_node1, 2)};
        const auto R1_data = Kokkos::Array<double, 4>{
            node_u(i_node1, 3), node_u(i_node1, 4), node_u(i_node1, 5), node_u(i_node1, 6)};
        const auto u1 = View_3::const_type{u1_data.data()};
        const auto R1 = Kokkos::View<double[4]>::const_type{R1_data.data()};

        // Target node displacement
        const auto R2_data = Kokkos::Array<double, 4>{
            node_u(i_node2, 3), node_u(i_node2, 4), node_u(i_node2, 5), node_u(i_node2, 6)};
        const auto R2 = Kokkos::View<double[4]>::const_type{R2_data.data()};
        const auto u2_data =
            Kokkos::Array<double, 3>{node_u(i_node2, 0), node_u(i_node2, 1), node_u(i_node2, 2)};
        const auto u2 = View_3::const_type{u2_data.data()};

        // Rotation control
        auto RC_data = Kokkos::Array<double, 4>{};
        const auto RC = Kokkos::View<double[4]>{RC_data.data()};
        auto RCt_data = Kokkos::Array<double, 4>{};
        const auto RCt = Kokkos::View<double[4]>{RCt_data.data()};
        auto RV_data = Kokkos::Array<double, 4>{};
        const auto RV = Kokkos::View<double[4]>{RV_data.data()};

        auto R1t_data = Kokkos::Array<double, 4>{};
        const auto R1t = Kokkos::View<double[4]>{R1t_data.data()};

        auto R1_X0_data = Kokkos::Array<double, 4>{};
        const auto R1_X0 = Kokkos::View<double[4]>{R1_X0_data.data()};

        auto R2_RCt_data = Kokkos::Array<double, 4>{};
        const auto R2_RCt = Kokkos::View<double[4]>{R2_RCt_data.data()};

        auto R2_RCt_R1t_data = Kokkos::Array<double, 4>{};
        const auto R2_RCt_R1t = Kokkos::View<double[4]>{R2_RCt_R1t_data.data()};

        auto A_data = Kokkos::Array<double, 9>{};
        const auto A = View_3x3{A_data.data()};

        auto C_data = Kokkos::Array<double, 9>{};
        const auto C = View_3x3{C_data.data()};

        auto V3_data = Kokkos::Array<double, 3>{};
        const auto V3 = View_3{V3_data.data()};

        //----------------------------------------------------------------------
        // Residual Vector
        //----------------------------------------------------------------------

        // Extract residual rows relevant to this constraint
        const auto Phi = Kokkos::subview(Phi_, row_range(i_constraint));

        // Phi(0:3) = u2 + X0 - u1 - R1*X0
        QuaternionInverse(R1, R1t);
        RotateVectorByQuaternion(R1, X0, R1_X0);
        for (int i = 0; i < 3; ++i) {
            Phi(i) = u2(i) + X0(i) - u1(i) - R1_X0(i);
        }

        // Angular residual
        // If this is a rotation control constraint, calculate RC from control and axis
        for (auto i = 0U; i < 3U; ++i) {
            RV(i) = axes(i_constraint, 0, i) * control(i_constraint);
        }
        RotationVectorToQuaternion(RV, RC);
        QuaternionInverse(RC, RCt);

        // Phi(3:6) = axial(R2*inv(RC)*inv(R1))
        QuaternionCompose(R2, RCt, R2_RCt);
        QuaternionCompose(R2_RCt, R1t, R2_RCt_R1t);
        QuaternionToRotationMatrix(R2_RCt_R1t, C);
        AxialVectorOfMatrix(C, V3);
        for (int i = 0; i < 3; ++i) {
            Phi(i + 3) = V3(i);
        }

        //----------------------------------------------------------------------
        // Constraint Gradient Matrix
        //----------------------------------------------------------------------

        //---------------------------------
        // Target Node
        //---------------------------------
        {
            // Extract gradient block for target node of this constraint
            const auto B = Kokkos::subview(gradient_terms, i_constraint, Kokkos::ALL, target_node_col_range(i_constraint));

            // B(0:3,0:3) = I
            for (int i = 0; i < 3; ++i) {
                B(i, i) = 1.;
            }

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
        {
            // Extract gradient block for base node of this constraint
            const auto B = Kokkos::subview(gradient_terms, i_constraint, Kokkos::ALL, base_node_col_range(i_constraint));

            // B(0:3,0:3) = -I
            for (int i = 0; i < 3; ++i) {
                B(i, i) = -1.;
            }

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
