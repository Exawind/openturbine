#pragma once

#include <array>
#include <vector>

#include <Kokkos_Core.hpp>

#include "constraint.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

/// @brief Constraints struct holds all constraint data
/// @details Constraints struct holds all constraint data and provides methods
/// to update the prescribed displacements and control signals.
/// @note The struct is used to transfer data between the solver and the
/// constraint objects.
struct Constraints {
    /// @brief DeviceData struct holds constraint data on device
    struct DeviceData {
        ConstraintType type;               //< Constraint type
        Kokkos::pair<int, int> row_range;  //< Range of rows in the global stiffness matrix
        int base_node_index;               //< Base node index
        int target_node_index;             //< Target node index
        double X0[3];                      //< Initial relative location between nodes
        double axis_x[3];                  // Unit vector representing x rotation axis
        double axis_y[3];                  // Unit vector representing y rotation axis
        double axis_z[3];                  // Unit vector representing z rotation axis
    };

    /// @brief HostData struct holds constraint data on host
    struct HostData {
        ConstraintType type;  //< Constraint type
        Array_7 u;            //< Prescribed displacement
        float* control;       //< Control signal
    };

    int num;                                       //< Number of constraints
    int num_dofs;                                  //< Number of degrees of freedom
    std::vector<HostData> constraint_data;         //< Host constraint data
    Kokkos::View<DeviceData*> data;                //< Device constraint data
    View_N control;                                //< Control signals
    View_Nx7 u;                                    //< Prescribed displacements
    View_N Phi;                                    //< Residual vector
    Kokkos::View<double* [6][12]> gradient_terms;  //< Gradient terms

    Constraints() = default;

    Constraints(const std::vector<std::shared_ptr<Constraint>>& constraints) {
        num = constraints.size();
        num_dofs = std::transform_reduce(
            constraints.cbegin(), constraints.cend(), 0, std::plus{},
            [](auto c) {
                return c->NumDOFs();
            }
        );
        constraint_data = std::vector<HostData>(
            num, HostData{ConstraintType::kNone, {0., 0., 0., 1., 0., 0., 0.}, nullptr}
        );
        data = Kokkos::View<DeviceData*>("data", num);
        control = View_N("control", num);
        u = View_Nx7("u", num);
        Phi = View_N("residual_vector", num_dofs);
        gradient_terms = Kokkos::View<double* [6][12]>("gradient_terms", num);

        // Create host mirror for constraint data
        auto host_data = Kokkos::create_mirror(this->data);

        // Loop through constraint input and set data
        int start_row = 0;
        for (int i = 0; i < this->num; ++i) {
            // Set Host constraint data
            this->constraint_data[i].type = constraints[i]->type;
            this->constraint_data[i].control = constraints[i]->control;

            // Set constraint type
            host_data(i).type = constraints[i]->type;

            // Set constraint rows
            auto dofs = constraints[i]->NumDOFs();
            host_data(i).row_range = Kokkos::make_pair(start_row, start_row + dofs);
            start_row += dofs;

            // Set base node and target node index
            host_data(i).base_node_index = constraints[i]->base_node.ID;
            host_data(i).target_node_index = constraints[i]->target_node.ID;

            // Set initial relative location between nodes
            host_data(i).X0[0] = constraints[i]->X0[0];
            host_data(i).X0[1] = constraints[i]->X0[1];
            host_data(i).X0[2] = constraints[i]->X0[2];

            // Set rotation axes
            for (int j = 0; j < 3; ++j) {
                host_data(i).axis_x[j] = constraints[i]->x_axis[j];
                host_data(i).axis_y[j] = constraints[i]->y_axis[j];
                host_data(i).axis_z[j] = constraints[i]->z_axis[j];
            }
        }

        // Update data
        Kokkos::deep_copy(this->data, host_data);

        // Set initial rotation to identity
        Kokkos::deep_copy(Kokkos::subview(this->u, Kokkos::ALL, 3), 1.0);
    }

    /// Sets the new displacement for the given constraint
    void UpdateDisplacement(int id, Array_7 u_) { this->constraint_data[id].u = u_; }

    /// Transfers new prescribed displacements and control signals to Views
    void UpdateViews() {
        auto host_u_mirror = Kokkos::create_mirror(this->u);
        auto host_control_mirror = Kokkos::create_mirror(this->control);
        for (size_t i = 0; i < this->constraint_data.size(); ++i) {
            switch (this->constraint_data[i].type) {
                case ConstraintType::kPrescribedBC: {
                    for (int j = 0; j < kLieGroupComponents; ++j) {
                        host_u_mirror(i, j) = this->constraint_data[i].u[j];
                    }
                } break;
                case ConstraintType::kRotationControl: {
                    host_control_mirror(i) = *this->constraint_data[i].control;
                } break;
                default:
                    break;
            }
        }
        Kokkos::deep_copy(this->control, host_control_mirror);
        Kokkos::deep_copy(this->u, host_u_mirror);
    }
};

}  // namespace openturbine
