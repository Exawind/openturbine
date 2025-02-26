#pragma once

#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
#include <KokkosBatched_Copy_Decl.hpp>

#include "calculate_fixed_bc_constraint.hpp"
#include "calculate_fixed_bc_3DOF_constraint.hpp"
#include "calculate_prescribed_bc_constraint.hpp"
#include "calculate_prescribed_bc_3DOF_constraint.hpp"
#include "calculate_revolute_joint_constraint.hpp"
#include "calculate_rigid_joint_constraint.hpp"
#include "calculate_rigid_joint_3DOF_constraint.hpp"
#include "calculate_rotation_control_constraint.hpp"
#include "constraints.hpp"
#include "math/quaternion_operations.hpp"
#include "math/vector_operations.hpp"

namespace openturbine {

struct CalculateConstraintResidualGradient {
    Kokkos::View<ConstraintType*>::const_type type;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type target_node_col_range;
    Kokkos::View<size_t*>::const_type base_node_index;
    Kokkos::View<size_t*>::const_type target_node_index;
    View_Nx3::const_type X0_;
    View_Nx3x3::const_type axes;
    View_Nx7::const_type constraint_inputs;
    Kokkos::View<double* [6]>::const_type lambda;
    Kokkos::View<double* [6][6]>::const_type tangent;
    View_Nx7::const_type node_u;
    View_Nx6 residual_terms;
    Kokkos::View<double* [6]> base_lambda_residual_terms;
    Kokkos::View<double* [6]> target_lambda_residual_terms;
    View_Nx6x6 base_gradient_terms;
    View_Nx6x6 target_gradient_terms;
    View_Nx6x6 base_gradient_transpose_terms;
    View_Nx6x6 target_gradient_transpose_terms;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        auto constraint_type = type(i_constraint);
        if (constraint_type == ConstraintType::kFixedBC) {
            CalculateFixedBCConstraint{
                i_constraint, target_node_col_range, target_node_index,     X0_, constraint_inputs,
                node_u,       residual_terms,        target_gradient_terms,
            }();
        }
        else if (constraint_type == ConstraintType::kFixedBC3DOFs) {
            CalculateFixedBC3DOFConstraint{
                i_constraint, target_node_col_range, target_node_index,     X0_, constraint_inputs,
                node_u,       residual_terms,        target_gradient_terms,
            }();
        }
        else if (constraint_type == ConstraintType::kPrescribedBC) {
            CalculatePrescribedBCConstraint{
                i_constraint, target_node_col_range, target_node_index,     X0_, constraint_inputs,
                node_u,       residual_terms,        target_gradient_terms,
            }();
        }
        else if (constraint_type == ConstraintType::kPrescribedBC3DOFs) {
            CalculatePrescribedBC3DOFConstraint{
                i_constraint, target_node_col_range, target_node_index,     X0_, constraint_inputs,
                node_u,       residual_terms,        target_gradient_terms,
            }();
        }
        else if (constraint_type == ConstraintType::kRigidJoint) {
            CalculateRigidJointConstraint{
                i_constraint,
                target_node_col_range,
                base_node_index,
                target_node_index,
                X0_,
                constraint_inputs,
                node_u,
                residual_terms,
                base_gradient_terms,
                target_gradient_terms,
            }();
        }
        else if (constraint_type == ConstraintType::kRigidJoint6DOFsTo3DOFs) {
            CalculateRigidJoint3DOFConstraint{
                i_constraint,
                target_node_col_range,
                base_node_index,
                target_node_index,
                X0_,
                constraint_inputs,
                node_u,
                residual_terms,
                base_gradient_terms,
                target_gradient_terms,
            }();
        }
        else if (constraint_type == ConstraintType::kRevoluteJoint) {
            CalculateRevoluteJointConstraint{
                i_constraint,
                base_node_index,
                target_node_index,
                X0_,
                axes,
                constraint_inputs,
                node_u,
                residual_terms,
                base_gradient_terms,
                target_gradient_terms,
            }();
        }
        else if (constraint_type == ConstraintType::kRotationControl) {
            CalculateRotationControlConstraint{
                i_constraint,
                base_node_index,
                target_node_index,
                X0_,
                axes,
                constraint_inputs,
                node_u,
                residual_terms,
                base_gradient_terms,
                target_gradient_terms,
            }();
        }

        KokkosBatched::SerialCopy<KokkosBatched::Trans::Transpose>::invoke(Kokkos::subview(base_gradient_terms, i_constraint, Kokkos::ALL, Kokkos::ALL), Kokkos::subview(base_gradient_transpose_terms, i_constraint, Kokkos::ALL, Kokkos::ALL));
        KokkosBatched::SerialCopy<KokkosBatched::Trans::Transpose>::invoke(Kokkos::subview(target_gradient_terms, i_constraint, Kokkos::ALL, Kokkos::ALL), Kokkos::subview(target_gradient_transpose_terms, i_constraint, Kokkos::ALL, Kokkos::ALL));

        {
            auto G_data = Kokkos::Array<double, 36>{};
            auto T_data = Kokkos::Array<double, 36>{};
            auto GT_data = Kokkos::Array<double, 36>{};

            const auto G = Kokkos::View<double[6][6]>(G_data.data());
            const auto T = Kokkos::View<double[6][6]>(T_data.data());
            const auto GT = Kokkos::View<double[6][6]>(GT_data.data());

            KokkosBatched::SerialCopy<>::invoke(Kokkos::subview(base_gradient_terms, i_constraint, Kokkos::ALL, Kokkos::ALL), G);
            KokkosBatched::SerialCopy<>::invoke(Kokkos::subview(tangent, base_node_index(i_constraint), Kokkos::ALL, Kokkos::ALL), T);

            KokkosBatched::SerialGemm<KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::NoTranspose, KokkosBatched::Algo::Gemm::Default>::invoke(1., G, T, 0., GT);

            KokkosBatched::SerialCopy<>::invoke(GT, Kokkos::subview(base_gradient_terms, i_constraint, Kokkos::ALL, Kokkos::ALL));
        }

        {
            auto G_data = Kokkos::Array<double, 36>{};
            auto T_data = Kokkos::Array<double, 36>{};
            auto GT_data = Kokkos::Array<double, 36>{};

            const auto G = Kokkos::View<double[6][6]>(G_data.data());
            const auto T = Kokkos::View<double[6][6]>(T_data.data());
            const auto GT = Kokkos::View<double[6][6]>(GT_data.data());

            KokkosBatched::SerialCopy<>::invoke(Kokkos::subview(target_gradient_terms, i_constraint, Kokkos::ALL, Kokkos::ALL), G);
            KokkosBatched::SerialCopy<>::invoke(Kokkos::subview(tangent, target_node_index(i_constraint), Kokkos::ALL, Kokkos::ALL), T);

            KokkosBatched::SerialGemm<KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::NoTranspose, KokkosBatched::Algo::Gemm::Default>::invoke(1., G, T, 0., GT);

            KokkosBatched::SerialCopy<>::invoke(GT, Kokkos::subview(target_gradient_terms, i_constraint, Kokkos::ALL, Kokkos::ALL));
        }

        {
            auto lr_data = Kokkos::Array<double, 6>{};
            auto B_t_data = Kokkos::Array<double, 36>{};
            auto l_data = Kokkos::Array<double, 6>{};

            const auto B_t = Kokkos::View<double[6][6]>(B_t_data.data());
            const auto l = Kokkos::View<double[6]>(l_data.data());
            const auto lr = Kokkos::View<double[6]>(lr_data.data());

            KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose, 1>::invoke(Kokkos::subview(lambda, i_constraint, Kokkos::ALL), l);

            KokkosBatched::SerialCopy<>::invoke(Kokkos::subview(base_gradient_transpose_terms, i_constraint, Kokkos::ALL, Kokkos::ALL), B_t);

            KokkosBlas::Experimental::serial_gemv('N', 1., B_t, l, 0., lr);

            KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose, 1>::invoke(lr, Kokkos::subview(base_lambda_residual_terms, i_constraint, Kokkos::ALL));

            KokkosBatched::SerialCopy<>::invoke(Kokkos::subview(target_gradient_transpose_terms, i_constraint, Kokkos::ALL, Kokkos::ALL), B_t);

            KokkosBlas::Experimental::serial_gemv('N', 1., B_t, l, 0., lr);

            KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose, 1>::invoke(lr, Kokkos::subview(target_lambda_residual_terms, i_constraint, Kokkos::ALL));
        }
    };
};

} // namespace openturbine
