#pragma once

#include <array>
#include <numeric>
#include <vector>

#include <Kokkos_Core.hpp>

#include "constraint.hpp"
#include "dof_management/freedom_signature.hpp"

namespace openturbine {

/**
 * @brief Container class for managing multiple constraints in a simulation
 *
 * @details Manages a collection of constraints between nodes, including their properties,
 * freedom signatures, and computational data structures. This class handles both single-node
 * boundary conditions and two-node constraints (like joints). It provides facilities for:
 * - Storing constraint properties (types, node indices, axes)
 * - Managing degrees of freedom and freedom signatures
 * - Handling control signals and prescribed displacements
 * - Maintaining computational views for residuals and gradients
 */
template <typename DeviceType>
struct Constraints {
    size_t num_constraints;  //< Total number of constraints in the system
    size_t num_dofs;         //< Total number of degrees of freedom controlled by constraints

    // Constraint properties
    Kokkos::View<ConstraintType*, DeviceType> type;       //< Type of each constraint
    std::vector<double*> control_signal;      //< Control signal for each constraint
    Kokkos::View<size_t*, DeviceType> base_node_index;    //< Index of the base node for each constraint
    Kokkos::View<size_t*, DeviceType> target_node_index;  //< Index of the target node for each constraint

    // DOF management
    Kokkos::View<Kokkos::pair<size_t, size_t>*, DeviceType> row_range;
    Kokkos::View<FreedomSignature*, DeviceType> base_node_freedom_signature;
    Kokkos::View<FreedomSignature*, DeviceType> target_node_freedom_signature;
    Kokkos::View<size_t*, DeviceType> base_active_dofs;
    Kokkos::View<size_t*, DeviceType> target_active_dofs;
    Kokkos::View<size_t* [6], DeviceType> base_node_freedom_table;
    Kokkos::View<size_t* [6], DeviceType> target_node_freedom_table;

    // Geometric configuration
    Kokkos::View<double* [3], DeviceType> X0;       //< Initial relative position
    Kokkos::View<double* [3][3], DeviceType> axes;  //< Rotation axes

    // State variables
    Kokkos::View<double* [7], DeviceType> input;   //< Current state input
    Kokkos::View<double* [3], DeviceType> output;  //< Current state output
    Kokkos::View<double* [6], DeviceType> lambda;

    // Host mirrors for CPU access
    typename Kokkos::View<double* [7], DeviceType>::HostMirror host_input;
    typename Kokkos::View<double* [3], DeviceType>::HostMirror host_output;

    // System contributions
    Kokkos::View<double* [6], DeviceType> residual_terms;
    Kokkos::View<double* [6], DeviceType> base_lambda_residual_terms;
    Kokkos::View<double* [6], DeviceType> target_lambda_residual_terms;
    Kokkos::View<double* [6], DeviceType> system_residual_terms;
    Kokkos::View<double* [6][6], DeviceType> base_gradient_terms;
    Kokkos::View<double* [6][6], DeviceType> target_gradient_terms;
    Kokkos::View<double* [6][6], DeviceType> base_gradient_transpose_terms;
    Kokkos::View<double* [6][6], DeviceType> target_gradient_transpose_terms;

    explicit Constraints(const std::vector<Constraint>& constraints, const std::vector<Node>& nodes)
        : num_constraints{constraints.size()},
          num_dofs{std::transform_reduce(
              constraints.cbegin(), constraints.cend(), 0U, std::plus{},
              [](auto c) {
                  return NumRowsForConstraint(c.type);
              }
          )},
          type(Kokkos::view_alloc("type", Kokkos::WithoutInitializing), num_constraints),
          control_signal(num_constraints),
          base_node_index(
              Kokkos::view_alloc("base_node_index", Kokkos::WithoutInitializing), num_constraints
          ),
          target_node_index(
              Kokkos::view_alloc("target_node_index", Kokkos::WithoutInitializing), num_constraints
          ),
          row_range(Kokkos::view_alloc("row_range", Kokkos::WithoutInitializing), num_constraints),
          base_node_freedom_signature(
              Kokkos::view_alloc("base_node_freedom_signature", Kokkos::WithoutInitializing),
              num_constraints
          ),
          target_node_freedom_signature(
              Kokkos::view_alloc("target_node_freedom_signature", Kokkos::WithoutInitializing),
              num_constraints
          ),
          base_active_dofs(
              Kokkos::view_alloc("base_active_dofs", Kokkos::WithoutInitializing), num_constraints
          ),
          target_active_dofs(
              Kokkos::view_alloc("target_active_dofs", Kokkos::WithoutInitializing), num_constraints
          ),
          base_node_freedom_table(
              Kokkos::view_alloc("base_node_freedom_table", Kokkos::WithoutInitializing),
              num_constraints
          ),
          target_node_freedom_table(
              Kokkos::view_alloc("target_node_freedom_table", Kokkos::WithoutInitializing),
              num_constraints
          ),
          X0(Kokkos::view_alloc("X0", Kokkos::WithoutInitializing), num_constraints),
          axes(Kokkos::view_alloc("axes", Kokkos::WithoutInitializing), num_constraints),
          input(Kokkos::view_alloc("inputs", Kokkos::WithoutInitializing), num_constraints),
          output(Kokkos::view_alloc("outputs", Kokkos::WithoutInitializing), num_constraints),
          lambda(Kokkos::view_alloc("lambda", Kokkos::WithoutInitializing), num_constraints),
          host_input(Kokkos::view_alloc("host_input", Kokkos::WithoutInitializing), num_constraints),
          host_output(
              Kokkos::view_alloc("host_output", Kokkos::WithoutInitializing), num_constraints
          ),
          residual_terms(
              Kokkos::view_alloc("residual_terms", Kokkos::WithoutInitializing), num_constraints
          ),
          base_lambda_residual_terms(
              Kokkos::view_alloc("base_lambda_residual_terms", Kokkos::WithoutInitializing),
              num_constraints
          ),
          target_lambda_residual_terms(
              Kokkos::view_alloc("target_lambda_residual_terms", Kokkos::WithoutInitializing),
              num_constraints
          ),
          system_residual_terms(
              Kokkos::view_alloc("system_residual_terms", Kokkos::WithoutInitializing),
              num_constraints
          ),
          base_gradient_terms(
              Kokkos::view_alloc("base_gradient_terms", Kokkos::WithoutInitializing), num_constraints
          ),
          target_gradient_terms(
              Kokkos::view_alloc("target_gradient_terms", Kokkos::WithoutInitializing),
              num_constraints
          ),
          base_gradient_transpose_terms(
              Kokkos::view_alloc("base_gradient_transpose_terms", Kokkos::WithoutInitializing),
              num_constraints
          ),
          target_gradient_transpose_terms(
              Kokkos::view_alloc("target_gradient_transpose_terms", Kokkos::WithoutInitializing),
              num_constraints
          ) {
        auto host_type = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, type);
        auto host_row_range = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, row_range);
        auto host_base_node_index =
            Kokkos::create_mirror_view(Kokkos::WithoutInitializing, base_node_index);
        auto host_target_node_index =
            Kokkos::create_mirror_view(Kokkos::WithoutInitializing, target_node_index);
        auto host_base_freedom =
            Kokkos::create_mirror_view(Kokkos::WithoutInitializing, base_node_freedom_signature);
        auto host_target_freedom =
            Kokkos::create_mirror_view(Kokkos::WithoutInitializing, target_node_freedom_signature);
        auto host_base_active_dofs =
            Kokkos::create_mirror_view(Kokkos::WithoutInitializing, base_active_dofs);
        auto host_target_active_dofs =
            Kokkos::create_mirror_view(Kokkos::WithoutInitializing, target_active_dofs);
        auto host_X0 = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, X0);
        auto host_axes = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, axes);

        auto start_row = size_t{0U};
        for (auto i = 0U; i < num_constraints; ++i) {
            const auto& c = constraints[i];
            const auto base_node_id = c.node_ids[0];
            const auto target_node_id = c.node_ids[1];

            // Set constraint properties
            host_type(i) = c.type;

            // Set the freedom signature from the constraint types
            if (c.type == ConstraintType::kFixedBC || c.type == ConstraintType::kPrescribedBC) {
                host_base_freedom(i) = FreedomSignature::NoComponents;
                host_target_freedom(i) = FreedomSignature::AllComponents;

                host_base_active_dofs(i) = 0UL;
                host_target_active_dofs(i) = 6UL;
            } else if (c.type == ConstraintType::kRigidJoint ||
                       c.type == ConstraintType::kRevoluteJoint ||
                       c.type == ConstraintType::kRotationControl) {
                host_base_freedom(i) = FreedomSignature::AllComponents;
                host_target_freedom(i) = FreedomSignature::AllComponents;

                host_base_active_dofs(i) = 6UL;
                host_target_active_dofs(i) = 6UL;
            } else if (c.type == ConstraintType::kFixedBC3DOFs ||
                       c.type == ConstraintType::kPrescribedBC3DOFs) {
                host_base_freedom(i) = FreedomSignature::NoComponents;
                host_target_freedom(i) = FreedomSignature::JustPosition;

                host_base_active_dofs(i) = 0UL;
                host_target_active_dofs(i) = 3UL;
            } else if (c.type == ConstraintType::kRigidJoint6DOFsTo3DOFs) {
                host_base_freedom(i) = FreedomSignature::AllComponents;
                host_target_freedom(i) = FreedomSignature::JustPosition;

                host_base_active_dofs(i) = 6UL;
                host_target_active_dofs(i) = 3UL;
            }

            control_signal[i] = c.control;

            // Set base and target node index
            host_base_node_index(i) = base_node_id;
            host_target_node_index(i) = target_node_id;

            // Set constraint rows
            auto n_rows = NumRowsForConstraint(host_type(i));
            host_row_range(i) = Kokkos::make_pair(start_row, start_row + n_rows);
            start_row += n_rows;

            // Calculate initial relative position (X0)
            Array_3 x0{0., 0., 0.};
            if (c.type != ConstraintType::kPrescribedBC &&
                c.type != ConstraintType::kPrescribedBC3DOFs) {
                x0 = CalculateX0(c, nodes[target_node_id], nodes[base_node_id]);
            }
            for (size_t j = 0; j < 3; ++j) {
                host_X0(i, j) = x0[j];
            }

            // Calculate rotation axes
            const auto rotation_matrix = CalculateAxes(c, x0);
            for (size_t j = 0; j < 3; ++j) {
                for (size_t k = 0; k < 3; ++k) {
                    host_axes(i, j, k) = rotation_matrix[j][k];
                }
            }

            // Initialize displacement to zero if prescribed BC
            if (c.type == ConstraintType::kPrescribedBC ||
                c.type == ConstraintType::kPrescribedBC3DOFs) {
                host_input(i, 0) = 0.;
                host_input(i, 1) = 0.;
                host_input(i, 2) = 0.;
                host_input(i, 3) = 1.;
                host_input(i, 4) = 0.;
                host_input(i, 5) = 0.;
                host_input(i, 6) = 0.;
            }
        }

        Kokkos::deep_copy(type, host_type);
        Kokkos::deep_copy(row_range, host_row_range);
        Kokkos::deep_copy(base_node_index, host_base_node_index);
        Kokkos::deep_copy(target_node_index, host_target_node_index);
        Kokkos::deep_copy(base_node_freedom_signature, host_base_freedom);
        Kokkos::deep_copy(target_node_freedom_signature, host_target_freedom);
        Kokkos::deep_copy(base_active_dofs, host_base_active_dofs);
        Kokkos::deep_copy(target_active_dofs, host_target_active_dofs);
        Kokkos::deep_copy(X0, host_X0);
        Kokkos::deep_copy(axes, host_axes);

        Kokkos::deep_copy(Kokkos::subview(this->input, Kokkos::ALL, 3), 1.);
    }

    /// Calculates the initial relative position (X0) based on constraint type and nodes
    static Array_3 CalculateX0(
        const Constraint& constraint, const Node& target_node, const Node& base_node
    ) {
        Array_3 x0{0., 0., 0.};
        // Set X0 to the prescribed displacement for fixed and prescribed BCs i.e. constraints
        // with 1 node
        if (GetNumberOfNodes(constraint.type) == 1) {
            x0[0] = target_node.x[0] - constraint.vec[0];
            x0[1] = target_node.x[1] - constraint.vec[1];
            x0[2] = target_node.x[2] - constraint.vec[2];
            return x0;
        }

        // Default: set X0 to the relative position between nodes
        x0[0] = target_node.x[0] - base_node.x[0];
        x0[1] = target_node.x[1] - base_node.x[1];
        x0[2] = target_node.x[2] - base_node.x[2];
        return x0;
    }

    /// Calculates the rotation axes for a constraint based on its type and configuration
    static Array_3x3 CalculateAxes(const Constraint& constraint, const Array_3& x0) {
        Array_3x3 rotation_matrix{};
        if (constraint.type == ConstraintType::kRevoluteJoint) {
            constexpr Array_3 x = {1., 0., 0.};
            const Array_3 x_hat =
                Norm(constraint.vec) != 0. ? UnitVector(constraint.vec) : UnitVector(x0);

            // Create rotation matrix to rotate x to match vector
            const auto cross_product = CrossProduct(x, x_hat);
            const auto dot_product = DotProduct(x_hat, x);
            const auto k = 1. / (1. + dot_product);

            // Set orthogonal unit vectors from the rotation matrix
            rotation_matrix[0][0] = cross_product[0] * cross_product[0] * k + dot_product;
            rotation_matrix[0][1] = cross_product[1] * cross_product[0] * k + cross_product[2];
            rotation_matrix[0][2] = cross_product[2] * cross_product[0] * k - cross_product[1];

            rotation_matrix[1][0] = cross_product[0] * cross_product[1] * k - cross_product[2];
            rotation_matrix[1][1] = cross_product[1] * cross_product[1] * k + dot_product;
            rotation_matrix[1][2] = cross_product[2] * cross_product[1] * k + cross_product[0];

            rotation_matrix[2][0] = cross_product[0] * cross_product[2] * k + cross_product[1];
            rotation_matrix[2][1] = cross_product[1] * cross_product[2] * k - cross_product[0];
            rotation_matrix[2][2] = cross_product[2] * cross_product[2] * k + dot_product;

            return rotation_matrix;
        }

        // Set rotation_matrix to the unit vector of the constraint axis for rotation control
        if (constraint.type == ConstraintType::kRotationControl) {
            const auto unit_vector = UnitVector(constraint.vec);
            rotation_matrix[0][0] = unit_vector[0];
            rotation_matrix[0][1] = unit_vector[1];
            rotation_matrix[0][2] = unit_vector[2];
            return rotation_matrix;
        }

        // If not a revolute/hinge joint, set rotation_matrix to the input vector
        rotation_matrix[0][0] = constraint.vec[0];
        rotation_matrix[0][1] = constraint.vec[1];
        rotation_matrix[0][2] = constraint.vec[2];
        return rotation_matrix;
    }

    /// Sets the new displacement for the given constraint
    void UpdateDisplacement(size_t constraint_id, const std::array<double, 7>& disp) const {
        for (auto i = 0U; i < 7U; ++i) {
            host_input(constraint_id, i) = disp[i];
        }
    }

    /// Transfers new prescribed displacements and control signals to Views
    void UpdateViews() {
        for (auto i = 0U; i < this->num_constraints; ++i) {
            if (control_signal[i] != nullptr) {
                host_input(i, 0) = *control_signal[i];
            }
        }
        Kokkos::deep_copy(this->input, host_input);
    }
};

}  // namespace openturbine
