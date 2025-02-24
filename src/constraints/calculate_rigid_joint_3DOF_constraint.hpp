#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"
#include "math/matrix_operations.hpp"
#include "math/quaternion_operations.hpp"
#include "math/vector_operations.hpp"

namespace openturbine {

struct CalculateRigidJoint3DOFConstraint {
    int i_constraint;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type target_node_col_range;
    Kokkos::View<size_t*>::const_type base_node_index;
    Kokkos::View<size_t*>::const_type target_node_index;

    View_Nx3::const_type X0_;
    View_Nx7::const_type constraint_inputs;
    View_Nx7::const_type node_u;
    View_Nx6 residual_terms;
    View_Nx6x6 base_gradient_terms;
    View_Nx6x6 target_gradient_terms;

    KOKKOS_FUNCTION
    static void CalculateResidualTerms(
        const int constraint_index,
        const View_Quaternion::const_type& R1,
        const View_3::const_type& X0, const View_3::const_type& u1, const View_3::const_type& u2,
        const View_3& R1_X0, const View_Nx6& residual
    ) {
        auto R1t_data = Kokkos::Array<double, 4>{};
        const auto R1t = View_Quaternion{R1t_data.data()};

        auto R2_R1t_data = Kokkos::Array<double, 4>{};
        auto R2_R1t = View_Quaternion{R2_R1t_data.data()};

        auto V3_data = Kokkos::Array<double, 3>{};
        auto V3 = View_3{V3_data.data()};

        // Phi(0:3) = u2 + X0 - u1 - R1*X0
        QuaternionInverse(R1, R1t);
        RotateVectorByQuaternion(R1, X0, R1_X0);
        for (int i = 0; i < 3; ++i) {
            residual(constraint_index, i) = u2(i) + X0(i) - u1(i) - R1_X0(i);
        }
    }

    KOKKOS_FUNCTION
    static void CalculateGradientTerms(
        const int constraint_index, const View_3::const_type& R1_X0,
        const View_Nx6x6& target_gradient, const View_Nx6x6& base_gradient
    ) {
        auto A_data = Kokkos::Array<double, 9>{};
        auto A = View_3x3{A_data.data()};

        // Target Node gradients
        {
            // B(0:3,0:3) = I
            for (int i = 0; i < 3; ++i) {
                target_gradient(constraint_index, i, i) = 1.;
            }
        }

        // Base Node gradients
        {
            // B(0:3,0:3) = -I
            for (int i = 0; i < 3; ++i) {
                base_gradient(constraint_index, i, i) = -1.;
            }

            // B(0:3,3:6) = tilde(R1*X0)
            VecTilde(R1_X0, A);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    base_gradient(constraint_index, i, j + 3) = A(i, j);
                }
            }
        }
    }

    KOKKOS_FUNCTION
    void operator()() const {
        const auto i_node1 = base_node_index(i_constraint);
        const auto i_node2 = target_node_index(i_constraint);

        // Initial difference between nodes
        const auto X0_data = Kokkos::Array<double, 3>{
            X0_(i_constraint, 0), X0_(i_constraint, 1), X0_(i_constraint, 2)
        };
        const auto X0 = View_3::const_type{X0_data.data()};

        // Base node translational displacement
        const auto u1_data =
            Kokkos::Array<double, 3>{node_u(i_node1, 0), node_u(i_node1, 1), node_u(i_node1, 2)};
        const auto u1 = View_3::const_type{u1_data.data()};

        // Base node rotational displacement
        const auto R1_data = Kokkos::Array<double, 4>{
            node_u(i_node1, 3), node_u(i_node1, 4), node_u(i_node1, 5), node_u(i_node1, 6)
        };
        const auto R1 = Kokkos::View<double[4]>::const_type{R1_data.data()};

        // Target node translational displacement
        const auto u2_data =
            Kokkos::Array<double, 3>{node_u(i_node2, 0), node_u(i_node2, 1), node_u(i_node2, 2)};
        const auto u2 = View_3::const_type{u2_data.data()};

        auto R1_X0_data = Kokkos::Array<double, 3>{};
        auto R1_X0 = Kokkos::View<double[3]>{R1_X0_data.data()};

        //----------------------------------------------------------------------
        // Residual Vector
        //----------------------------------------------------------------------

        CalculateResidualTerms(
            i_constraint, R1, X0, u1, u2, R1_X0, residual_terms
        );

        //----------------------------------------------------------------------
        // Constraint Gradient Matrix
        //----------------------------------------------------------------------

        CalculateGradientTerms(
            i_constraint, R1_X0, target_gradient_terms, base_gradient_terms
        );
    }
};

}  // namespace openturbine
