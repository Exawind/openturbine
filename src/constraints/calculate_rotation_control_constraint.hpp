#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"

#include "src/math/quaternion_operations.hpp"
#include "src/math/vector_operations.hpp"

namespace openturbine {

struct CalculateRotationControlConstraint {
    Kokkos::View<size_t*>::const_type base_node_index;
    Kokkos::View<size_t*>::const_type target_node_index;
    Kokkos::View<double* [3]>::const_type X0_;
    Kokkos::View<double* [3][3]>::const_type axes;
    Kokkos::View<double* [7]>::const_type constraint_inputs;
    Kokkos::View<double* [7]>::const_type node_u;
    Kokkos::View<double* [6]> residual_terms;
    Kokkos::View<double* [6][6]> base_gradient_terms;
    Kokkos::View<double* [6][6]> target_gradient_terms;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        const auto i_b = base_node_index(i_constraint);
        const auto i_t = target_node_index(i_constraint);

        // Initial difference between nodes
        const auto X0_data = Kokkos::Array<double, 3>{
            X0_(i_constraint, 0), X0_(i_constraint, 1), X0_(i_constraint, 2)
        };
        const auto X0 = View_3::const_type{X0_data.data()};

        // Base node displacement (translation and rotation)
        const auto ub_data =
            Kokkos::Array<double, 3>{node_u(i_b, 0), node_u(i_b, 1), node_u(i_b, 2)};
        const auto Rb_data =
            Kokkos::Array<double, 4>{node_u(i_b, 3), node_u(i_b, 4), node_u(i_b, 5), node_u(i_b, 6)};
        const auto ub = View_3::const_type{ub_data.data()};
        const auto Rb = Kokkos::View<double[4]>::const_type{Rb_data.data()};

        // Target node displacement (translation and rotation)
        const auto ut_data =
            Kokkos::Array<double, 3>{node_u(i_t, 0), node_u(i_t, 1), node_u(i_t, 2)};
        const auto ut = View_3::const_type{ut_data.data()};
        const auto Rt_data =
            Kokkos::Array<double, 4>{node_u(i_t, 3), node_u(i_t, 4), node_u(i_t, 5), node_u(i_t, 6)};
        const auto Rt = Kokkos::View<double[4]>::const_type{Rt_data.data()};

        auto AX_data = Kokkos::Array<double, 3>{};
        const auto AX = Kokkos::View<double[3]>{AX_data.data()};

        // Control rotation vector
        auto RV_data = Kokkos::Array<double, 3>{};
        const auto RV = Kokkos::View<double[3]>{RV_data.data()};

        // Rotation control
        auto Rc_data = Kokkos::Array<double, 4>{};
        const auto Rc = Kokkos::View<double[4]>{Rc_data.data()};
        auto RcT_data = Kokkos::Array<double, 4>{};
        const auto RcT = Kokkos::View<double[4]>{RcT_data.data()};

        // Base rotation transpose
        auto RbT_data = Kokkos::Array<double, 4>{};
        const auto RbT = Kokkos::View<double[4]>{RbT_data.data()};

        // Base rotation * X0
        auto Rb_X0_data = Kokkos::Array<double, 4>{};
        const auto Rb_X0 = Kokkos::View<double[4]>{Rb_X0_data.data()};

        // Target rotation * Control rotation transpose
        auto Rt_RcT_data = Kokkos::Array<double, 4>{};
        const auto Rt_RcT = Kokkos::View<double[4]>{Rt_RcT_data.data()};

        // Target rotation * Control rotation transpose * Base rotation transpose
        auto Rt_RcT_RbT_data = Kokkos::Array<double, 4>{};
        const auto Rt_RcT_RbT = Kokkos::View<double[4]>{Rt_RcT_RbT_data.data()};

        auto A_data = Kokkos::Array<double, 9>{};
        const auto A = View_3x3{A_data.data()};

        auto C_data = Kokkos::Array<double, 9>{};
        const auto C = View_3x3{C_data.data()};

        auto V3_data = Kokkos::Array<double, 3>{};
        const auto V3 = View_3{V3_data.data()};

        //----------------------------------------------------------------------
        // Position residual
        //----------------------------------------------------------------------

        // Phi(0:3) = ut + X0 - ub - Rb*X0
        RotateVectorByQuaternion(Rb, X0, Rb_X0);
        for (int i = 0; i < 3; ++i) {
            residual_terms(i_constraint, i) = ut(i) + X0(i) - ub(i) - Rb_X0(i);
        }

        //----------------------------------------------------------------------
        // Rotation residual
        //----------------------------------------------------------------------

        auto rotation_command = constraint_inputs(i_constraint, 0);

        // Copy rotation axis for this constraint
        for (auto i = 0U; i < 3U; ++i) {
            AX(i) = axes(i_constraint, 0, i);
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
        for (auto i = 0U; i < 3U; ++i) {
            residual_terms(i_constraint, i + 3) = V3(i);
        }

        //----------------------------------------------------------------------
        // Constraint Gradient Matrix
        //----------------------------------------------------------------------

        //---------------------------------
        // Target Node
        //---------------------------------
        {
            // B(0:3,0:3) = I
            for (int i = 0; i < 3; ++i) {
                target_gradient_terms(i_constraint, i, i) = 1.;
            }

            // B(3:6,3:6) = AX(Rb*Rc*inv(Rt)) = transpose(AX(Rt*inv(Rc)*inv(Rb)))
            AX_Matrix(C, A);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    target_gradient_terms(i_constraint, i + 3, j + 3) = A(j, i);
                }
            }
        }
        //---------------------------------
        // Base Node
        //---------------------------------
        {
            // B(0:3,0:3) = -I
            for (int i = 0; i < 3; ++i) {
                base_gradient_terms(i_constraint, i, i) = -1.;
            }

            // B(0:3,3:6) = tilde(Rb*X0)
            VecTilde(Rb_X0, A);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    base_gradient_terms(i_constraint, i, j + 3) = A(i, j);
                }
            }

            // B(3:6,3:6) = -AX(Rt*inv(Rc)*inv(Rb))
            AX_Matrix(C, A);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    base_gradient_terms(i_constraint, i + 3, j + 3) = -A(i, j);
                }
            }
        }
    }
};

}  // namespace openturbine
