#pragma once

#include <array>
#include <numeric>
#include <vector>

#include <Kokkos_Core.hpp>

#include "constraint.hpp"
#include "dof_management/freedom_signature.hpp"
#include "model/node.hpp"

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
    template <typename ValueType>
    using View = Kokkos::View<ValueType, DeviceType>;
    template <typename ValueType>
    using ConstView = typename View<ValueType>::const_type;

    size_t num_constraints;  //< Total number of constraints in the system
    size_t num_dofs;         //< Total number of degrees of freedom controlled by constraints

    // Constraint properties
    View<constraints::ConstraintType*> type;  //< Type of each constraint
    std::vector<double*> control_signal;      //< Control signal for each constraint
    View<size_t*> base_node_index;            //< Index of the base node for each constraint
    View<size_t*> target_node_index;          //< Index of the target node for each constraint

    // DOF management
    View<Kokkos::pair<size_t, size_t>*> row_range;
    View<dof::FreedomSignature*> base_node_freedom_signature;
    View<dof::FreedomSignature*> target_node_freedom_signature;
    View<size_t*> base_active_dofs;
    View<size_t*> target_active_dofs;
    View<size_t* [6]> base_node_freedom_table;
    View<size_t* [6]> target_node_freedom_table;

    // Geometric configuration
    View<double* [3]> X0;       //< Initial relative position
    View<double* [3][3]> axes;  //< Rotation axes

    // State variables
    View<double* [7]> input;   //< Current state input
    View<double* [3]> output;  //< Current state output
    View<double* [6]> lambda;

    // Host mirrors for CPU access
    typename View<double* [7]>::HostMirror host_input;
    typename View<double* [3]>::HostMirror host_output;

    // System contributions
    View<double* [6]> residual_terms;
    View<double* [6]> base_lambda_residual_terms;
    View<double* [6]> target_lambda_residual_terms;
    View<double* [6]> system_residual_terms;
    View<double* [6][6]> base_gradient_terms;
    View<double* [6][6]> target_gradient_terms;
    View<double* [6][6]> base_gradient_transpose_terms;
    View<double* [6][6]> target_gradient_transpose_terms;

    explicit Constraints(
        const std::vector<constraints::Constraint>& constraints, const std::vector<Node>& nodes
    )
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
        using Kokkos::ALL;
        using Kokkos::create_mirror_view;
        using Kokkos::deep_copy;
        using Kokkos::subview;
        using Kokkos::WithoutInitializing;

        auto host_type = create_mirror_view(WithoutInitializing, type);
        auto host_row_range = create_mirror_view(WithoutInitializing, row_range);
        auto host_base_node_index = create_mirror_view(WithoutInitializing, base_node_index);
        auto host_target_node_index = create_mirror_view(WithoutInitializing, target_node_index);
        auto host_base_freedom =
            create_mirror_view(WithoutInitializing, base_node_freedom_signature);
        auto host_target_freedom =
            create_mirror_view(WithoutInitializing, target_node_freedom_signature);
        auto host_base_active_dofs = create_mirror_view(WithoutInitializing, base_active_dofs);
        auto host_target_active_dofs = create_mirror_view(WithoutInitializing, target_active_dofs);
        auto host_X0 = create_mirror_view(WithoutInitializing, X0);
        auto host_axes = create_mirror_view(WithoutInitializing, axes);

        auto start_row = size_t{0U};
        for (auto constraint = 0U; constraint < num_constraints; ++constraint) {
            const auto& c = constraints[constraint];
            const auto base_node_id = c.node_ids[0];
            const auto target_node_id = c.node_ids[1];

            // Set constraint properties
            host_type(constraint) = c.type;

            // Set the freedom signature from the constraint types
            if (c.type == constraints::ConstraintType::FixedBC ||
                c.type == constraints::ConstraintType::PrescribedBC) {
                host_base_freedom(constraint) = dof::FreedomSignature::NoComponents;
                host_target_freedom(constraint) = dof::FreedomSignature::AllComponents;

                host_base_active_dofs(constraint) = 0UL;
                host_target_active_dofs(constraint) = 6UL;
            } else if (c.type == constraints::ConstraintType::RigidJoint ||
                       c.type == constraints::ConstraintType::RevoluteJoint ||
                       c.type == constraints::ConstraintType::RotationControl) {
                host_base_freedom(constraint) = dof::FreedomSignature::AllComponents;
                host_target_freedom(constraint) = dof::FreedomSignature::AllComponents;

                host_base_active_dofs(constraint) = 6UL;
                host_target_active_dofs(constraint) = 6UL;
            } else if (c.type == constraints::ConstraintType::FixedBC3DOFs ||
                       c.type == constraints::ConstraintType::PrescribedBC3DOFs) {
                host_base_freedom(constraint) = dof::FreedomSignature::NoComponents;
                host_target_freedom(constraint) = dof::FreedomSignature::JustPosition;

                host_base_active_dofs(constraint) = 0UL;
                host_target_active_dofs(constraint) = 3UL;
            } else if (c.type == constraints::ConstraintType::RigidJoint6DOFsTo3DOFs) {
                host_base_freedom(constraint) = dof::FreedomSignature::AllComponents;
                host_target_freedom(constraint) = dof::FreedomSignature::JustPosition;

                host_base_active_dofs(constraint) = 6UL;
                host_target_active_dofs(constraint) = 3UL;
            }

            control_signal[constraint] = c.control;

            // Set base and target node index
            host_base_node_index(constraint) = base_node_id;
            host_target_node_index(constraint) = target_node_id;

            // Set constraint rows
            auto n_rows = NumRowsForConstraint(host_type(constraint));
            host_row_range(constraint) = Kokkos::make_pair(start_row, start_row + n_rows);
            start_row += n_rows;

            // Calculate initial relative position (X0)
            std::array<double, 3> x0{0., 0., 0.};
            if (c.type != constraints::ConstraintType::PrescribedBC &&
                c.type != constraints::ConstraintType::PrescribedBC3DOFs) {
                x0 = CalculateX0(c, nodes[target_node_id], nodes[base_node_id]);
            }
            for (auto component = 0U; component < 3U; ++component) {
                host_X0(constraint, component) = x0[component];
            }

            // Calculate rotation axes
            const auto rotation_matrix = CalculateAxes(c, x0);
            for (auto component_1 = 0U; component_1 < 3U; ++component_1) {
                for (auto component_2 = 0U; component_2 < 3U; ++component_2) {
                    host_axes(constraint, component_1, component_2) =
                        rotation_matrix[component_1][component_2];
                }
            }

            // Initialize displacement to provided displacement if prescribed BC
            if (c.type == constraints::ConstraintType::PrescribedBC ||
                c.type == constraints::ConstraintType::PrescribedBC3DOFs) {
                for (auto component = 0U; component < 7U; ++component) {
                    host_input(constraint, component) = c.initial_displacement[component];
                }
            }
        }

        deep_copy(type, host_type);
        deep_copy(row_range, host_row_range);
        deep_copy(base_node_index, host_base_node_index);
        deep_copy(target_node_index, host_target_node_index);
        deep_copy(base_node_freedom_signature, host_base_freedom);
        deep_copy(target_node_freedom_signature, host_target_freedom);
        deep_copy(base_active_dofs, host_base_active_dofs);
        deep_copy(target_active_dofs, host_target_active_dofs);
        deep_copy(X0, host_X0);
        deep_copy(axes, host_axes);

        deep_copy(subview(this->input, ALL, 3), 1.);
    }

    /// Calculates the initial relative position (X0) based on constraint type and nodes
    static std::array<double, 3> CalculateX0(
        const constraints::Constraint& constraint, const Node& target_node, const Node& base_node
    ) {
        std::array<double, 3> x0{0., 0., 0.};
        // Set X0 to the prescribed displacement for fixed and prescribed BCs i.e. constraints
        // with 1 node
        if (GetNumberOfNodes(constraint.type) == 1) {
            x0[0] = target_node.x0[0] - constraint.axis_vector[0];
            x0[1] = target_node.x0[1] - constraint.axis_vector[1];
            x0[2] = target_node.x0[2] - constraint.axis_vector[2];
            return x0;
        }

        // Default: set X0 to the relative position between nodes
        x0[0] = target_node.x0[0] - base_node.x0[0];
        x0[1] = target_node.x0[1] - base_node.x0[1];
        x0[2] = target_node.x0[2] - base_node.x0[2];
        return x0;
    }

    /// Calculates the rotation axes for a constraint based on its type and configuration
    static std::array<std::array<double, 3>, 3> CalculateAxes(
        const constraints::Constraint& constraint, const std::array<double, 3>& x0
    ) {
        std::array<std::array<double, 3>, 3> rotation_matrix{};
        if (constraint.type == constraints::ConstraintType::RevoluteJoint) {
            constexpr std::array<double, 3> x = {1., 0., 0.};
            const std::array<double, 3> x_hat = math::Norm(constraint.axis_vector) != 0.
                                                    ? math::UnitVector(constraint.axis_vector)
                                                    : math::UnitVector(x0);

            // Create rotation matrix to rotate x to match vector
            const auto cross_product = math::CrossProduct(x, x_hat);
            const auto dot_product = math::DotProduct(x_hat, x);
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
        if (constraint.type == constraints::ConstraintType::RotationControl) {
            const auto unit_vector = math::UnitVector(constraint.axis_vector);
            rotation_matrix[0][0] = unit_vector[0];
            rotation_matrix[0][1] = unit_vector[1];
            rotation_matrix[0][2] = unit_vector[2];
            return rotation_matrix;
        }

        // If not a revolute/hinge joint, set rotation_matrix to the input vector
        rotation_matrix[0][0] = constraint.axis_vector[0];
        rotation_matrix[0][1] = constraint.axis_vector[1];
        rotation_matrix[0][2] = constraint.axis_vector[2];
        return rotation_matrix;
    }

    /// Sets the new displacement for the given constraint
    void UpdateDisplacement(size_t constraint_id, const std::array<double, 7>& disp) const {
        for (auto component = 0U; component < 7U; ++component) {
            host_input(constraint_id, component) = disp[component];
        }
    }

    /// Transfers new prescribed displacements and control signals to Views
    void UpdateViews() {
        for (auto constraint = 0U; constraint < this->num_constraints; ++constraint) {
            if (control_signal[constraint] != nullptr) {
                host_input(constraint, 0) = *control_signal[constraint];
            }
        }
        Kokkos::deep_copy(this->input, host_input);
    }
};

}  // namespace openturbine
