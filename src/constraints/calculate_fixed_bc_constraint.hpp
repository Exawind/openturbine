#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"

#include "src/math/matrix_operations.hpp"
#include "src/math/quaternion_operations.hpp"
#include "src/math/vector_operations.hpp"

namespace openturbine {

struct CalculateFixedBCConstraint {
    int i_constraint;
    Kokkos::View<size_t*>::const_type target_node_index;
    Kokkos::View<double* [3]>::const_type X0_;
    Kokkos::View<double* [7]>::const_type constraint_inputs;
    Kokkos::View<double* [7]>::const_type node_u;
    Kokkos::View<double* [6]> residual_terms;
    Kokkos::View<double* [6][6]> target_gradient_terms;

    KOKKOS_FUNCTION
    void operator()() const {
        const auto i_node2 = target_node_index(i_constraint);

        // Initial difference between nodes
        const auto X0_data = Kokkos::Array<double, 3>{
            X0_(i_constraint, 0), X0_(i_constraint, 1), X0_(i_constraint, 2)
        };
        const auto X0 = View_3::const_type{X0_data.data()};

        // Base node displacement
        constexpr auto u1_data = Kokkos::Array<double, 3>{0., 0., 0.};
        const auto u1 = View_3::const_type{u1_data.data()};
        constexpr auto R1_data = Kokkos::Array<double, 4>{1., 0., 0., 0.};
        const auto R1 = Kokkos::View<double[4]>::const_type{R1_data.data()};

        // Target node displacement
        const auto R2_data = Kokkos::Array<double, 4>{
            node_u(i_node2, 3), node_u(i_node2, 4), node_u(i_node2, 5), node_u(i_node2, 6)
        };
        const auto R2 = Kokkos::View<double[4]>::const_type{R2_data.data()};
        const auto u2_data =
            Kokkos::Array<double, 3>{node_u(i_node2, 0), node_u(i_node2, 1), node_u(i_node2, 2)};
        const auto u2 = View_3::const_type{u2_data.data()};

        // Rotation control
        constexpr auto RCt_data = Kokkos::Array<double, 4>{1., 0., 0., 0.};
        const auto RCt = Kokkos::View<double[4]>::const_type{RCt_data.data()};

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

        // Phi(0:3) = u2 + X0 - u1 - R1*X0
        QuaternionInverse(R1, R1t);
        RotateVectorByQuaternion(R1, X0, R1_X0);
        for (int i = 0; i < 3; ++i) {
            residual_terms(i_constraint, i) = u2(i) + X0(i) - u1(i) - R1_X0(i);
        }

        // Angular residual
        // Phi(3:6) = axial(R2*inv(RC)*inv(R1))
        QuaternionCompose(R2, RCt, R2_RCt);
        QuaternionCompose(R2_RCt, R1t, R2_RCt_R1t);
        QuaternionToRotationMatrix(R2_RCt_R1t, C);
        AxialVectorOfMatrix(C, V3);
        for (int i = 0; i < 3; ++i) {
            residual_terms(i_constraint, i + 3) = V3(i);
        }

        //----------------------------------------------------------------------
        // Constraint Gradient Matrix
        //----------------------------------------------------------------------

        //---------------------------------
        // Target Node
        //---------------------------------

        // B(0:3,0:3) = I
        for (int i = 0; i < 3; ++i) {
            target_gradient_terms(i_constraint, i, i) = 1.;
        }

        // B(3:6,3:6) = AX(R1*RC*inv(R2)) = transpose(AX(R2*inv(RC)*inv(R1)))
        AX_Matrix(C, A);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                target_gradient_terms(i_constraint, i + 3, j + 3) = A(j, i);
            }
        }
    }
};

}  // namespace openturbine
