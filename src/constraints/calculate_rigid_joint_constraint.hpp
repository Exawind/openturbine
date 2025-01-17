#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"

#include "src/math/matrix_operations.hpp"
#include "src/math/quaternion_operations.hpp"
#include "src/math/vector_operations.hpp"

namespace openturbine {

struct CalculateRigidJointConstraint {
    int i_constraint;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type target_node_col_range;
    Kokkos::View<size_t*>::const_type base_node_index;
    Kokkos::View<size_t*>::const_type target_node_index;
    Kokkos::View<double* [3]>::const_type X0_;
    Kokkos::View<double* [7]>::const_type constraint_inputs;
    Kokkos::View<double* [7]>::const_type node_u;
    Kokkos::View<double* [6]> residual_terms;
    Kokkos::View<double* [6][6]> base_gradient_terms;
    Kokkos::View<double* [6][6]> target_gradient_terms;

    KOKKOS_FUNCTION
    static void CalculateResidualTerms(
        const int constraint_index, const size_t n_target_node_cols,
        const Kokkos::View<double[4]>::const_type& R1, const Kokkos::View<double[4]>::const_type& R2,
        const Kokkos::View<double[4]>::const_type& RCt, const View_3::const_type& X0,
        const View_3::const_type& u1, const View_3::const_type& u2,
        const Kokkos::View<double[4]>& R1t, const Kokkos::View<double[4]>& R1_X0,
        const Kokkos::View<double[4]>& R2_RCt, const Kokkos::View<double[4]>& R2_RCt_R1t,
        const View_3x3& C, const View_3& V3, const Kokkos::View<double* [6]>& residual
    ) {
        // Phi(0:3) = u2 + X0 - u1 - R1*X0
        QuaternionInverse(R1, R1t);
        RotateVectorByQuaternion(R1, X0, R1_X0);
        for (int i = 0; i < 3; ++i) {
            residual(constraint_index, i) = u2(i) + X0(i) - u1(i) - R1_X0(i);
        }

        // Angular residual
        for (int i = 0; i < 3; ++i) {
            residual(constraint_index, i + 3) = 0.;
        }
        // Phi(3:6) = axial(R2*inv(RC)*inv(R1))
        if (n_target_node_cols == 6) {
            QuaternionCompose(R2, RCt, R2_RCt);
            QuaternionCompose(R2_RCt, R1t, R2_RCt_R1t);
            QuaternionToRotationMatrix(R2_RCt_R1t, C);
            AxialVectorOfMatrix(C, V3);
            for (int i = 0; i < 3; ++i) {
                residual(constraint_index, i + 3) = V3(i);
            }
        }
    }

    KOKKOS_FUNCTION
    static void CalculateGradientTerms(
        const int constraint_index, const size_t n_target_node_cols,
        const Kokkos::View<double[4]>::const_type& R1_X0, const View_3x3& C, const View_3x3& A,
        const Kokkos::View<double* [6][6]>& target_gradient,
        const Kokkos::View<double* [6][6]>& base_gradient
    ) {
        // Target Node gradients
        {
            // B(0:3,0:3) = I
            for (int i = 0; i < 3; ++i) {
                target_gradient(constraint_index, i, i) = 1.;
            }

            // B(3:6,3:6) -> initialize to 0
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    target_gradient(constraint_index, i + 3, j + 3) = 0.;
                }
            }
            // B(3:6,3:6) = AX(R1*RC*inv(R2)) = transpose(AX(R2*inv(RC)*inv(R1)))
            if (n_target_node_cols == 6) {
                AX_Matrix(C, A);
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        target_gradient(constraint_index, i + 3, j + 3) = A(j, i);
                    }
                }
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

            // B(3:6,3:6) -> initialize to 0
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    base_gradient(constraint_index, i + 3, j + 3) = 0.;
                }
            }
            // B(3:6,3:6) = -AX(R2*inv(RC)*inv(R1))
            if (n_target_node_cols == 6) {
                AX_Matrix(C, A);
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        base_gradient(constraint_index, i + 3, j + 3) = -A(i, j);
                    }
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

        // Base node displacement
        const auto u1_data =
            Kokkos::Array<double, 3>{node_u(i_node1, 0), node_u(i_node1, 1), node_u(i_node1, 2)};
        const auto u1 = View_3::const_type{u1_data.data()};
        const auto R1_data = Kokkos::Array<double, 4>{
            node_u(i_node1, 3), node_u(i_node1, 4), node_u(i_node1, 5), node_u(i_node1, 6)
        };
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
        auto R1_X0 = Kokkos::View<double[4]>{R1_X0_data.data()};

        auto R2_RCt_data = Kokkos::Array<double, 4>{};
        auto R2_RCt = Kokkos::View<double[4]>{R2_RCt_data.data()};

        auto R2_RCt_R1t_data = Kokkos::Array<double, 4>{};
        auto R2_RCt_R1t = Kokkos::View<double[4]>{R2_RCt_R1t_data.data()};

        auto A_data = Kokkos::Array<double, 9>{};
        auto A = View_3x3{A_data.data()};

        auto C_data = Kokkos::Array<double, 9>{};
        auto C = View_3x3{C_data.data()};

        auto V3_data = Kokkos::Array<double, 3>{};
        auto V3 = View_3{V3_data.data()};

        //----------------------------------------------------------------------
        // Residual Vector
        //----------------------------------------------------------------------

        const auto n_target_node_cols =
            target_node_col_range(i_constraint).second - target_node_col_range(i_constraint).first;

        CalculateResidualTerms(
            i_constraint, n_target_node_cols, R1, R2, RCt, X0, u1, u2, R1t, R1_X0, R2_RCt,
            R2_RCt_R1t, C, V3, residual_terms
        );

        //----------------------------------------------------------------------
        // Constraint Gradient Matrix
        //----------------------------------------------------------------------

        CalculateGradientTerms(
            i_constraint, n_target_node_cols, R1_X0, C, A, target_gradient_terms, base_gradient_terms
        );
    }
};

}  // namespace openturbine
