#pragma once

#include <KokkosBatched_Copy_Decl.hpp>
#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "calculate_fixed_bc_3DOF_constraint.hpp"
#include "calculate_fixed_bc_constraint.hpp"
#include "calculate_prescribed_bc_3DOF_constraint.hpp"
#include "calculate_prescribed_bc_constraint.hpp"
#include "calculate_revolute_joint_constraint.hpp"
#include "calculate_revolute_joint_force.hpp"
#include "calculate_rigid_joint_3DOF_constraint.hpp"
#include "calculate_rigid_joint_constraint.hpp"
#include "calculate_rotation_control_constraint.hpp"
#include "constraints.hpp"
#include "math/quaternion_operations.hpp"
#include "math/vector_operations.hpp"

namespace openturbine {

struct CalculateConstraintResidualGradient {
    using MatrixTranspose = KokkosBatched::SerialCopy<KokkosBatched::Trans::Transpose>;
    using Gemm = KokkosBatched::SerialGemm<
        KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::NoTranspose,
        KokkosBatched::Algo::Gemm::Default>;
    using VectorCopy = KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose, 1>;
    using MatrixCopy = KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose, 2>;

    Kokkos::View<ConstraintType*>::const_type type_;
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type target_node_col_range_;
    Kokkos::View<size_t*>::const_type base_node_index_;
    Kokkos::View<size_t*>::const_type target_node_index_;
    Kokkos::View<double* [3]>::const_type X0_;
    Kokkos::View<double* [3][3]>::const_type axes_;
    Kokkos::View<double* [7]>::const_type constraint_inputs_;
    Kokkos::View<double* [6]>::const_type lambda_;
    Kokkos::View<double* [6][6]>::const_type tangent_;
    Kokkos::View<double* [7]>::const_type node_u_;
    Kokkos::View<double* [6]> res_;
    Kokkos::View<double* [6]> b_lambda_res_;
    Kokkos::View<double* [6]> t_lambda_res_;
    Kokkos::View<double* [6]> system_res_;
    Kokkos::View<double* [6][6]> b_grad_;
    Kokkos::View<double* [6][6]> t_grad_;
    Kokkos::View<double* [6][6]> b_grad_trans_;
    Kokkos::View<double* [6][6]> t_grad_trans_;

    KOKKOS_FUNCTION
    void FixedBC(size_t i) const {
        using Kokkos::ALL;
        using Kokkos::subview;
        using KokkosBlas::Experimental::serial_gemv;

        auto X0_data = Kokkos::Array<double, 3>{};
        auto t_node_u_data = Kokkos::Array<double, 7>{};
        auto lambda_data = Kokkos::Array<double, 6>{};
        auto res_data = Kokkos::Array<double, 6>{};
        auto t_lambda_res_data = Kokkos::Array<double, 6>{};
        auto t_grad_data = Kokkos::Array<double, 36>{};
        auto t_grad_trans_data = Kokkos::Array<double, 36>{};
        auto target_tangent_data = Kokkos::Array<double, 36>{};
        auto t_grad_tan_data = Kokkos::Array<double, 36>{};

        const auto X0 = Kokkos::View<double[3]>(X0_data.data());
        const auto t_node_u = Kokkos::View<double[7]>(t_node_u_data.data());
        const auto lambda = Kokkos::View<double[6]>(lambda_data.data());
        const auto res = Kokkos::View<double[6]>(res_data.data());
        const auto t_lambda_res = Kokkos::View<double[6]>(t_lambda_res_data.data());
        const auto t_grad = Kokkos::View<double[6][6]>(t_grad_data.data());
        const auto t_grad_trans = Kokkos::View<double[6][6]>(t_grad_trans_data.data());
        const auto target_tangent = Kokkos::View<double[6][6]>(target_tangent_data.data());
        const auto t_grad_tan = Kokkos::View<double[6][6]>(t_grad_tan_data.data());

        const auto target_node_index = target_node_index_(i);

        VectorCopy::invoke(subview(X0_, i, ALL), X0);
        VectorCopy::invoke(subview(node_u_, target_node_index, ALL), t_node_u);
        VectorCopy::invoke(subview(lambda_, i, ALL), lambda);
        MatrixCopy::invoke(subview(tangent_, target_node_index, ALL, ALL), target_tangent);

        CalculateFixedBCConstraint(X0, t_node_u, res, t_grad);

        MatrixTranspose::invoke(t_grad, t_grad_trans);
        Gemm::invoke(1., t_grad, target_tangent, 0., t_grad_tan);
        serial_gemv('N', 1., t_grad_trans, lambda, 0., t_lambda_res);

        VectorCopy::invoke(res, subview(res_, i, ALL));
        VectorCopy::invoke(t_lambda_res, subview(t_lambda_res_, i, ALL));
        MatrixCopy::invoke(t_grad_tan, subview(t_grad_, i, ALL, ALL));
        MatrixCopy::invoke(t_grad_trans, subview(t_grad_trans_, i, ALL, ALL));
    }

    KOKKOS_FUNCTION
    void FixedBC3DOF(size_t i) const {
        using Kokkos::ALL;
        using Kokkos::subview;
        using KokkosBlas::Experimental::serial_gemv;

        auto X0_data = Kokkos::Array<double, 3>{};
        auto t_node_u_data = Kokkos::Array<double, 7>{};
        auto lambda_data = Kokkos::Array<double, 6>{};
        auto res_data = Kokkos::Array<double, 6>{};
        auto t_lambda_res_data = Kokkos::Array<double, 6>{};
        auto t_grad_data = Kokkos::Array<double, 36>{};
        auto t_grad_trans_data = Kokkos::Array<double, 36>{};
        auto target_tangent_data = Kokkos::Array<double, 36>{};
        auto t_grad_tan_data = Kokkos::Array<double, 36>{};

        const auto X0 = Kokkos::View<double[3]>(X0_data.data());
        const auto t_node_u = Kokkos::View<double[7]>(t_node_u_data.data());
        const auto lambda = Kokkos::View<double[6]>(lambda_data.data());
        const auto res = Kokkos::View<double[6]>(res_data.data());
        const auto t_lambda_res = Kokkos::View<double[6]>(t_lambda_res_data.data());
        const auto t_grad = Kokkos::View<double[6][6]>(t_grad_data.data());
        const auto t_grad_trans = Kokkos::View<double[6][6]>(t_grad_trans_data.data());
        const auto target_tangent = Kokkos::View<double[6][6]>(target_tangent_data.data());
        const auto t_grad_tan = Kokkos::View<double[6][6]>(t_grad_tan_data.data());

        const auto target_node_index = target_node_index_(i);

        VectorCopy::invoke(subview(X0_, i, ALL), X0);
        VectorCopy::invoke(subview(node_u_, target_node_index, ALL), t_node_u);
        VectorCopy::invoke(subview(lambda_, i, ALL), lambda);
        MatrixCopy::invoke(subview(tangent_, target_node_index, ALL, ALL), target_tangent);

        CalculateFixedBC3DOFConstraint(X0, t_node_u, res, t_grad);

        MatrixTranspose::invoke(t_grad, t_grad_trans);
        Gemm::invoke(1., t_grad, target_tangent, 0., t_grad_tan);
        serial_gemv('N', 1., t_grad_trans, lambda, 0., t_lambda_res);

        VectorCopy::invoke(res, subview(res_, i, ALL));
        VectorCopy::invoke(t_lambda_res, subview(t_lambda_res_, i, ALL));
        MatrixCopy::invoke(t_grad_tan, subview(t_grad_, i, ALL, ALL));
        MatrixCopy::invoke(t_grad_trans, subview(t_grad_trans_, i, ALL, ALL));
    }

    KOKKOS_FUNCTION
    void PrescribedBC(size_t i) const {
        using Kokkos::ALL;
        using Kokkos::subview;
        using KokkosBlas::Experimental::serial_gemv;

        auto X0_data = Kokkos::Array<double, 3>{};
        auto inputs_data = Kokkos::Array<double, 7>{};
        auto t_node_u_data = Kokkos::Array<double, 7>{};
        auto lambda_data = Kokkos::Array<double, 6>{};
        auto res_data = Kokkos::Array<double, 6>{};
        auto t_lambda_res_data = Kokkos::Array<double, 6>{};
        auto t_grad_data = Kokkos::Array<double, 36>{};
        auto t_grad_trans_data = Kokkos::Array<double, 36>{};
        auto target_tangent_data = Kokkos::Array<double, 36>{};
        auto t_grad_tan_data = Kokkos::Array<double, 36>{};

        const auto X0 = Kokkos::View<double[3]>(X0_data.data());
        const auto inputs = Kokkos::View<double[7]>(inputs_data.data());
        const auto t_node_u = Kokkos::View<double[7]>(t_node_u_data.data());
        const auto lambda = Kokkos::View<double[6]>(lambda_data.data());
        const auto res = Kokkos::View<double[6]>(res_data.data());
        const auto t_lambda_res = Kokkos::View<double[6]>(t_lambda_res_data.data());
        const auto t_grad = Kokkos::View<double[6][6]>(t_grad_data.data());
        const auto t_grad_trans = Kokkos::View<double[6][6]>(t_grad_trans_data.data());
        const auto target_tangent = Kokkos::View<double[6][6]>(target_tangent_data.data());
        const auto t_grad_tan = Kokkos::View<double[6][6]>(t_grad_tan_data.data());

        const auto target_node_index = target_node_index_(i);

        VectorCopy::invoke(subview(X0_, i, ALL), X0);
        VectorCopy::invoke(subview(constraint_inputs_, i, ALL), inputs);
        VectorCopy::invoke(subview(node_u_, target_node_index, ALL), t_node_u);
        VectorCopy::invoke(subview(lambda_, i, ALL), lambda);
        MatrixCopy::invoke(subview(tangent_, target_node_index, ALL, ALL), target_tangent);

        CalculatePrescribedBCConstraint(X0, inputs, t_node_u, res, t_grad);

        MatrixTranspose::invoke(t_grad, t_grad_trans);
        Gemm::invoke(1., t_grad, target_tangent, 0., t_grad_tan);
        serial_gemv('N', 1., t_grad_trans, lambda, 0., t_lambda_res);

        VectorCopy::invoke(res, subview(res_, i, ALL));
        VectorCopy::invoke(t_lambda_res, subview(t_lambda_res_, i, ALL));
        MatrixCopy::invoke(t_grad_tan, subview(t_grad_, i, ALL, ALL));
        MatrixCopy::invoke(t_grad_trans, subview(t_grad_trans_, i, ALL, ALL));
    }

    KOKKOS_FUNCTION
    void PrescribedBC3DOF(size_t i) const {
        using Kokkos::ALL;
        using Kokkos::subview;
        using KokkosBlas::Experimental::serial_gemv;

        auto X0_data = Kokkos::Array<double, 3>{};
        auto inputs_data = Kokkos::Array<double, 7>{};
        auto t_node_u_data = Kokkos::Array<double, 7>{};
        auto lambda_data = Kokkos::Array<double, 6>{};
        auto res_data = Kokkos::Array<double, 6>{};
        auto t_lambda_res_data = Kokkos::Array<double, 6>{};
        auto t_grad_data = Kokkos::Array<double, 36>{};
        auto t_grad_trans_data = Kokkos::Array<double, 36>{};
        auto target_tangent_data = Kokkos::Array<double, 36>{};
        auto t_grad_tan_data = Kokkos::Array<double, 36>{};

        const auto X0 = Kokkos::View<double[3]>(X0_data.data());
        const auto inputs = Kokkos::View<double[7]>(inputs_data.data());
        const auto t_node_u = Kokkos::View<double[7]>(t_node_u_data.data());
        const auto lambda = Kokkos::View<double[6]>(lambda_data.data());
        const auto res = Kokkos::View<double[6]>(res_data.data());
        const auto t_lambda_res = Kokkos::View<double[6]>(t_lambda_res_data.data());
        const auto t_grad = Kokkos::View<double[6][6]>(t_grad_data.data());
        const auto t_grad_trans = Kokkos::View<double[6][6]>(t_grad_trans_data.data());
        const auto target_tangent = Kokkos::View<double[6][6]>(target_tangent_data.data());
        const auto t_grad_tan = Kokkos::View<double[6][6]>(t_grad_tan_data.data());

        const auto target_node_index = target_node_index_(i);

        VectorCopy::invoke(subview(X0_, i, ALL), X0);
        VectorCopy::invoke(subview(constraint_inputs_, i, ALL), inputs);
        VectorCopy::invoke(subview(node_u_, target_node_index, ALL), t_node_u);
        VectorCopy::invoke(subview(lambda_, i, ALL), lambda);
        MatrixCopy::invoke(subview(tangent_, target_node_index, ALL, ALL), target_tangent);

        CalculatePrescribedBC3DOFConstraint(X0, inputs, t_node_u, res, t_grad);

        MatrixTranspose::invoke(t_grad, t_grad_trans);
        Gemm::invoke(1., t_grad, target_tangent, 0., t_grad_tan);
        serial_gemv('N', 1., t_grad_trans, lambda, 0., t_lambda_res);

        VectorCopy::invoke(res, subview(res_, i, ALL));
        VectorCopy::invoke(t_lambda_res, subview(t_lambda_res_, i, ALL));
        MatrixCopy::invoke(t_grad_tan, subview(t_grad_, i, ALL, ALL));
        MatrixCopy::invoke(t_grad_trans, subview(t_grad_trans_, i, ALL, ALL));
    }

    KOKKOS_FUNCTION
    void RigidJoint(size_t i) const {
        using Kokkos::ALL;
        using Kokkos::subview;
        using KokkosBlas::Experimental::serial_gemv;

        auto X0_data = Kokkos::Array<double, 3>{};
        auto b_node_u_data = Kokkos::Array<double, 7>{};
        auto t_node_u_data = Kokkos::Array<double, 7>{};
        auto lambda_data = Kokkos::Array<double, 6>{};
        auto res_data = Kokkos::Array<double, 6>{};
        auto b_lambda_res_data = Kokkos::Array<double, 6>{};
        auto t_lambda_res_data = Kokkos::Array<double, 6>{};
        auto b_grad_data = Kokkos::Array<double, 36>{};
        auto t_grad_data = Kokkos::Array<double, 36>{};
        auto b_grad_trans_data = Kokkos::Array<double, 36>{};
        auto t_grad_trans_data = Kokkos::Array<double, 36>{};
        auto base_tangent_data = Kokkos::Array<double, 36>{};
        auto target_tangent_data = Kokkos::Array<double, 36>{};
        auto b_grad_tan_data = Kokkos::Array<double, 36>{};
        auto t_grad_tan_data = Kokkos::Array<double, 36>{};

        const auto X0 = Kokkos::View<double[3]>(X0_data.data());
        const auto b_node_u = Kokkos::View<double[7]>(b_node_u_data.data());
        const auto t_node_u = Kokkos::View<double[7]>(t_node_u_data.data());
        const auto lambda = Kokkos::View<double[6]>(lambda_data.data());
        const auto res = Kokkos::View<double[6]>(res_data.data());
        const auto b_lambda_res = Kokkos::View<double[6]>(b_lambda_res_data.data());
        const auto t_lambda_res = Kokkos::View<double[6]>(t_lambda_res_data.data());
        const auto b_grad = Kokkos::View<double[6][6]>(b_grad_data.data());
        const auto t_grad = Kokkos::View<double[6][6]>(t_grad_data.data());
        const auto b_grad_trans = Kokkos::View<double[6][6]>(b_grad_trans_data.data());
        const auto t_grad_trans = Kokkos::View<double[6][6]>(t_grad_trans_data.data());
        const auto base_tangent = Kokkos::View<double[6][6]>(base_tangent_data.data());
        const auto target_tangent = Kokkos::View<double[6][6]>(target_tangent_data.data());
        const auto b_grad_tan = Kokkos::View<double[6][6]>(b_grad_tan_data.data());
        const auto t_grad_tan = Kokkos::View<double[6][6]>(t_grad_tan_data.data());

        const auto base_node_index = base_node_index_(i);
        const auto target_node_index = target_node_index_(i);

        VectorCopy::invoke(subview(X0_, i, ALL), X0);
        VectorCopy::invoke(subview(node_u_, base_node_index, ALL), b_node_u);
        VectorCopy::invoke(subview(node_u_, target_node_index, ALL), t_node_u);
        VectorCopy::invoke(subview(lambda_, i, ALL), lambda);
        MatrixCopy::invoke(subview(tangent_, base_node_index, ALL, ALL), base_tangent);
        MatrixCopy::invoke(subview(tangent_, target_node_index, ALL, ALL), target_tangent);

        CalculateRigidJointConstraint(X0, b_node_u, t_node_u, res, b_grad, t_grad);

        MatrixTranspose::invoke(b_grad, b_grad_trans);
        MatrixTranspose::invoke(t_grad, t_grad_trans);
        Gemm::invoke(1., b_grad, base_tangent, 0., b_grad_tan);
        Gemm::invoke(1., t_grad, target_tangent, 0., t_grad_tan);
        serial_gemv('N', 1., b_grad_trans, lambda, 0., b_lambda_res);
        serial_gemv('N', 1., t_grad_trans, lambda, 0., t_lambda_res);

        VectorCopy::invoke(res, subview(res_, i, ALL));
        VectorCopy::invoke(b_lambda_res, subview(b_lambda_res_, i, ALL));
        VectorCopy::invoke(t_lambda_res, subview(t_lambda_res_, i, ALL));
        MatrixCopy::invoke(b_grad_tan, subview(b_grad_, i, ALL, ALL));
        MatrixCopy::invoke(t_grad_tan, subview(t_grad_, i, ALL, ALL));
        MatrixCopy::invoke(b_grad_trans, subview(b_grad_trans_, i, ALL, ALL));
        MatrixCopy::invoke(t_grad_trans, subview(t_grad_trans_, i, ALL, ALL));
    }

    KOKKOS_FUNCTION
    void RigidJoint3DOF(size_t i) const {
        using Kokkos::ALL;
        using Kokkos::subview;
        using KokkosBlas::Experimental::serial_gemv;

        auto X0_data = Kokkos::Array<double, 3>{};
        auto b_node_u_data = Kokkos::Array<double, 7>{};
        auto t_node_u_data = Kokkos::Array<double, 7>{};
        auto lambda_data = Kokkos::Array<double, 6>{};
        auto res_data = Kokkos::Array<double, 6>{};
        auto b_lambda_res_data = Kokkos::Array<double, 6>{};
        auto t_lambda_res_data = Kokkos::Array<double, 6>{};
        auto b_grad_data = Kokkos::Array<double, 36>{};
        auto t_grad_data = Kokkos::Array<double, 36>{};
        auto b_grad_trans_data = Kokkos::Array<double, 36>{};
        auto t_grad_trans_data = Kokkos::Array<double, 36>{};
        auto base_tangent_data = Kokkos::Array<double, 36>{};
        auto target_tangent_data = Kokkos::Array<double, 36>{};
        auto b_grad_tan_data = Kokkos::Array<double, 36>{};
        auto t_grad_tan_data = Kokkos::Array<double, 36>{};

        const auto X0 = Kokkos::View<double[3]>(X0_data.data());
        const auto b_node_u = Kokkos::View<double[7]>(b_node_u_data.data());
        const auto t_node_u = Kokkos::View<double[7]>(t_node_u_data.data());
        const auto lambda = Kokkos::View<double[6]>(lambda_data.data());
        const auto res = Kokkos::View<double[6]>(res_data.data());
        const auto b_lambda_res = Kokkos::View<double[6]>(b_lambda_res_data.data());
        const auto t_lambda_res = Kokkos::View<double[6]>(t_lambda_res_data.data());
        const auto b_grad = Kokkos::View<double[6][6]>(b_grad_data.data());
        const auto t_grad = Kokkos::View<double[6][6]>(t_grad_data.data());
        const auto b_grad_trans = Kokkos::View<double[6][6]>(b_grad_trans_data.data());
        const auto t_grad_trans = Kokkos::View<double[6][6]>(t_grad_trans_data.data());
        const auto base_tangent = Kokkos::View<double[6][6]>(base_tangent_data.data());
        const auto target_tangent = Kokkos::View<double[6][6]>(target_tangent_data.data());
        const auto b_grad_tan = Kokkos::View<double[6][6]>(b_grad_tan_data.data());
        const auto t_grad_tan = Kokkos::View<double[6][6]>(t_grad_tan_data.data());

        const auto base_node_index = base_node_index_(i);
        const auto target_node_index = target_node_index_(i);

        VectorCopy::invoke(subview(X0_, i, ALL), X0);
        VectorCopy::invoke(subview(node_u_, base_node_index, ALL), b_node_u);
        VectorCopy::invoke(subview(node_u_, target_node_index, ALL), t_node_u);
        VectorCopy::invoke(subview(lambda_, i, ALL), lambda);
        MatrixCopy::invoke(subview(tangent_, base_node_index, ALL, ALL), base_tangent);
        MatrixCopy::invoke(subview(tangent_, target_node_index, ALL, ALL), target_tangent);

        CalculateRigidJoint3DOFConstraint(X0, b_node_u, t_node_u, res, b_grad, t_grad);

        MatrixTranspose::invoke(b_grad, b_grad_trans);
        MatrixTranspose::invoke(t_grad, t_grad_trans);
        Gemm::invoke(1., b_grad, base_tangent, 0., b_grad_tan);
        Gemm::invoke(1., t_grad, target_tangent, 0., t_grad_tan);
        serial_gemv('N', 1., b_grad_trans, lambda, 0., b_lambda_res);
        serial_gemv('N', 1., t_grad_trans, lambda, 0., t_lambda_res);

        VectorCopy::invoke(res, subview(res_, i, ALL));
        VectorCopy::invoke(b_lambda_res, subview(b_lambda_res_, i, ALL));
        VectorCopy::invoke(t_lambda_res, subview(t_lambda_res_, i, ALL));
        MatrixCopy::invoke(b_grad_tan, subview(b_grad_, i, ALL, ALL));
        MatrixCopy::invoke(t_grad_tan, subview(t_grad_, i, ALL, ALL));
        MatrixCopy::invoke(b_grad_trans, subview(b_grad_trans_, i, ALL, ALL));
        MatrixCopy::invoke(t_grad_trans, subview(t_grad_trans_, i, ALL, ALL));
    }

    KOKKOS_FUNCTION
    void RevoluteJoint(size_t i) const {
        using Kokkos::ALL;
        using Kokkos::subview;
        using KokkosBlas::Experimental::serial_gemv;

        auto X0_data = Kokkos::Array<double, 3>{};
        auto inputs_data = Kokkos::Array<double, 7>{};
        auto b_node_u_data = Kokkos::Array<double, 7>{};
        auto t_node_u_data = Kokkos::Array<double, 7>{};
        auto lambda_data = Kokkos::Array<double, 6>{};
        auto axes_data = Kokkos::Array<double, 9>{};
        auto res_data = Kokkos::Array<double, 6>{};
        auto b_lambda_res_data = Kokkos::Array<double, 6>{};
        auto t_lambda_res_data = Kokkos::Array<double, 6>{};
        auto system_res_data = Kokkos::Array<double, 6>{};
        auto b_grad_data = Kokkos::Array<double, 36>{};
        auto t_grad_data = Kokkos::Array<double, 36>{};
        auto b_grad_trans_data = Kokkos::Array<double, 36>{};
        auto t_grad_trans_data = Kokkos::Array<double, 36>{};
        auto base_tangent_data = Kokkos::Array<double, 36>{};
        auto target_tangent_data = Kokkos::Array<double, 36>{};
        auto b_grad_tan_data = Kokkos::Array<double, 36>{};
        auto t_grad_tan_data = Kokkos::Array<double, 36>{};

        const auto X0 = Kokkos::View<double[3]>(X0_data.data());
        const auto inputs = Kokkos::View<double[7]>(inputs_data.data());
        const auto b_node_u = Kokkos::View<double[7]>(b_node_u_data.data());
        const auto t_node_u = Kokkos::View<double[7]>(t_node_u_data.data());
        const auto lambda = Kokkos::View<double[6]>(lambda_data.data());
        const auto axes = Kokkos::View<double[3][3]>(axes_data.data());
        const auto res = Kokkos::View<double[6]>(res_data.data());
        const auto b_lambda_res = Kokkos::View<double[6]>(b_lambda_res_data.data());
        const auto t_lambda_res = Kokkos::View<double[6]>(t_lambda_res_data.data());
        const auto system_res = Kokkos::View<double[6]>(system_res_data.data());
        const auto b_grad = Kokkos::View<double[6][6]>(b_grad_data.data());
        const auto t_grad = Kokkos::View<double[6][6]>(t_grad_data.data());
        const auto b_grad_trans = Kokkos::View<double[6][6]>(b_grad_trans_data.data());
        const auto t_grad_trans = Kokkos::View<double[6][6]>(t_grad_trans_data.data());
        const auto base_tangent = Kokkos::View<double[6][6]>(base_tangent_data.data());
        const auto target_tangent = Kokkos::View<double[6][6]>(target_tangent_data.data());
        const auto b_grad_tan = Kokkos::View<double[6][6]>(b_grad_tan_data.data());
        const auto t_grad_tan = Kokkos::View<double[6][6]>(t_grad_tan_data.data());

        const auto base_node_index = base_node_index_(i);
        const auto target_node_index = target_node_index_(i);

        VectorCopy::invoke(subview(X0_, i, ALL), X0);
        VectorCopy::invoke(subview(constraint_inputs_, i, ALL), inputs);
        VectorCopy::invoke(subview(node_u_, base_node_index, ALL), b_node_u);
        VectorCopy::invoke(subview(node_u_, target_node_index, ALL), t_node_u);
        VectorCopy::invoke(subview(lambda_, i, ALL), lambda);
        MatrixCopy::invoke(subview(axes_, i, ALL, ALL), axes);
        MatrixCopy::invoke(subview(tangent_, base_node_index, ALL, ALL), base_tangent);
        MatrixCopy::invoke(subview(tangent_, target_node_index, ALL, ALL), target_tangent);

        CalculateRevoluteJointConstraint(X0, axes, b_node_u, t_node_u, res, b_grad, t_grad);
        CalculateRevoluteJointForce(axes, inputs, t_node_u, system_res);

        MatrixTranspose::invoke(b_grad, b_grad_trans);
        MatrixTranspose::invoke(t_grad, t_grad_trans);
        Gemm::invoke(1., b_grad, base_tangent, 0., b_grad_tan);
        Gemm::invoke(1., t_grad, target_tangent, 0., t_grad_tan);
        serial_gemv('N', 1., b_grad_trans, lambda, 0., b_lambda_res);
        serial_gemv('N', 1., t_grad_trans, lambda, 0., t_lambda_res);

        VectorCopy::invoke(res, subview(res_, i, ALL));
        VectorCopy::invoke(b_lambda_res, subview(b_lambda_res_, i, ALL));
        VectorCopy::invoke(t_lambda_res, subview(t_lambda_res_, i, ALL));
        VectorCopy::invoke(system_res, subview(system_res_, i, ALL));
        MatrixCopy::invoke(b_grad_tan, subview(b_grad_, i, ALL, ALL));
        MatrixCopy::invoke(t_grad_tan, subview(t_grad_, i, ALL, ALL));
        MatrixCopy::invoke(b_grad_trans, subview(b_grad_trans_, i, ALL, ALL));
        MatrixCopy::invoke(t_grad_trans, subview(t_grad_trans_, i, ALL, ALL));
    }

    KOKKOS_FUNCTION
    void RotationControl(size_t i) const {
        using Kokkos::ALL;
        using Kokkos::subview;
        using KokkosBlas::Experimental::serial_gemv;

        auto X0_data = Kokkos::Array<double, 3>{};
        auto inputs_data = Kokkos::Array<double, 7>{};
        auto b_node_u_data = Kokkos::Array<double, 7>{};
        auto t_node_u_data = Kokkos::Array<double, 7>{};
        auto lambda_data = Kokkos::Array<double, 6>{};
        auto axes_data = Kokkos::Array<double, 9>{};
        auto res_data = Kokkos::Array<double, 6>{};
        auto b_lambda_res_data = Kokkos::Array<double, 6>{};
        auto t_lambda_res_data = Kokkos::Array<double, 6>{};
        auto b_grad_data = Kokkos::Array<double, 36>{};
        auto t_grad_data = Kokkos::Array<double, 36>{};
        auto b_grad_trans_data = Kokkos::Array<double, 36>{};
        auto t_grad_trans_data = Kokkos::Array<double, 36>{};
        auto base_tangent_data = Kokkos::Array<double, 36>{};
        auto target_tangent_data = Kokkos::Array<double, 36>{};
        auto b_grad_tan_data = Kokkos::Array<double, 36>{};
        auto t_grad_tan_data = Kokkos::Array<double, 36>{};

        const auto X0 = Kokkos::View<double[3]>(X0_data.data());
        const auto inputs = Kokkos::View<double[7]>(inputs_data.data());
        const auto b_node_u = Kokkos::View<double[7]>(b_node_u_data.data());
        const auto t_node_u = Kokkos::View<double[7]>(t_node_u_data.data());
        const auto lambda = Kokkos::View<double[6]>(lambda_data.data());
        const auto axes = Kokkos::View<double[3][3]>(axes_data.data());
        const auto res = Kokkos::View<double[6]>(res_data.data());
        const auto b_lambda_res = Kokkos::View<double[6]>(b_lambda_res_data.data());
        const auto t_lambda_res = Kokkos::View<double[6]>(t_lambda_res_data.data());
        const auto b_grad = Kokkos::View<double[6][6]>(b_grad_data.data());
        const auto t_grad = Kokkos::View<double[6][6]>(t_grad_data.data());
        const auto b_grad_trans = Kokkos::View<double[6][6]>(b_grad_trans_data.data());
        const auto t_grad_trans = Kokkos::View<double[6][6]>(t_grad_trans_data.data());
        const auto base_tangent = Kokkos::View<double[6][6]>(base_tangent_data.data());
        const auto target_tangent = Kokkos::View<double[6][6]>(target_tangent_data.data());
        const auto b_grad_tan = Kokkos::View<double[6][6]>(b_grad_tan_data.data());
        const auto t_grad_tan = Kokkos::View<double[6][6]>(t_grad_tan_data.data());

        const auto base_node_index = base_node_index_(i);
        const auto target_node_index = target_node_index_(i);

        VectorCopy::invoke(subview(X0_, i, ALL), X0);
        VectorCopy::invoke(subview(constraint_inputs_, i, ALL), inputs);
        VectorCopy::invoke(subview(node_u_, base_node_index, ALL), b_node_u);
        VectorCopy::invoke(subview(node_u_, target_node_index, ALL), t_node_u);
        VectorCopy::invoke(subview(lambda_, i, ALL), lambda);
        MatrixCopy::invoke(subview(axes_, i, ALL, ALL), axes);
        MatrixCopy::invoke(subview(tangent_, base_node_index, ALL, ALL), base_tangent);
        MatrixCopy::invoke(subview(tangent_, target_node_index, ALL, ALL), target_tangent);

        CalculateRotationControlConstraint(
            X0, axes, inputs, b_node_u, t_node_u, res, b_grad, t_grad
        );

        MatrixTranspose::invoke(b_grad, b_grad_trans);
        MatrixTranspose::invoke(t_grad, t_grad_trans);
        Gemm::invoke(1., b_grad, base_tangent, 0., b_grad_tan);
        Gemm::invoke(1., t_grad, target_tangent, 0., t_grad_tan);
        serial_gemv('N', 1., b_grad_trans, lambda, 0., b_lambda_res);
        serial_gemv('N', 1., t_grad_trans, lambda, 0., t_lambda_res);

        VectorCopy::invoke(res, subview(res_, i, ALL));
        VectorCopy::invoke(b_lambda_res, subview(b_lambda_res_, i, ALL));
        VectorCopy::invoke(t_lambda_res, subview(t_lambda_res_, i, ALL));
        MatrixCopy::invoke(b_grad_tan, subview(b_grad_, i, ALL, ALL));
        MatrixCopy::invoke(t_grad_tan, subview(t_grad_, i, ALL, ALL));
        MatrixCopy::invoke(b_grad_trans, subview(b_grad_trans_, i, ALL, ALL));
        MatrixCopy::invoke(t_grad_trans, subview(t_grad_trans_, i, ALL, ALL));
    }

    KOKKOS_FUNCTION
    void operator()(size_t i) const {
        const auto constraint_type = type_(i);

        if (constraint_type == ConstraintType::kFixedBC) {
            FixedBC(i);
        } else if (constraint_type == ConstraintType::kFixedBC3DOFs) {
            FixedBC3DOF(i);
        } else if (constraint_type == ConstraintType::kPrescribedBC) {
            PrescribedBC(i);
        } else if (constraint_type == ConstraintType::kPrescribedBC3DOFs) {
            PrescribedBC3DOF(i);
        } else if (constraint_type == ConstraintType::kRigidJoint) {
            RigidJoint(i);
        } else if (constraint_type == ConstraintType::kRigidJoint6DOFsTo3DOFs) {
            RigidJoint3DOF(i);
        } else if (constraint_type == ConstraintType::kRevoluteJoint) {
            RevoluteJoint(i);
        } else if (constraint_type == ConstraintType::kRotationControl) {
            RotationControl(i);
        }
    };
};

}  // namespace openturbine
