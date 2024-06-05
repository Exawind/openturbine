#pragma once

#include <Kokkos_Core.hpp>

#include "constraints.hpp"

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct CalculateConstraintX0 {
    Kokkos::View<Constraints::Data*> data;
    Kokkos::View<int*>::const_type beam_node_state_indices;
    View_Nx7::const_type beam_node_x0;
    Kokkos::View<int*>::const_type mass_node_state_indices;
    View_Nx7::const_type mass_node_x0;

    KOKKOS_FUNCTION
    void operator()(const int i_constraint) const {
        auto& cd = data(i_constraint);
        bool node_found;
        double node1_x0[3];
        double node2_x0[3];

        node_found = false;
        for (int i = 0; i < beam_node_state_indices.extent_int(0); i++) {
            if (cd.target_node_index == beam_node_state_indices[i]) {
                node2_x0[0] = beam_node_x0(i, 0);
                node2_x0[1] = beam_node_x0(i, 1);
                node2_x0[2] = beam_node_x0(i, 2);
                node_found = true;
                break;
            }
        }
        if (!node_found) {
            for (int i = 0; i < mass_node_state_indices.extent_int(0); i++) {
                if (cd.target_node_index == mass_node_state_indices[i]) {
                    node2_x0[0] = mass_node_x0(i, 0);
                    node2_x0[1] = mass_node_x0(i, 1);
                    node2_x0[2] = mass_node_x0(i, 2);
                    break;
                }
            }
        }

        if (cd.type == ConstraintType::FixedBC || cd.type == ConstraintType::PrescribedBC) {
            cd.X0[0] = node2_x0[0] - cd.X0[0];
            cd.X0[1] = node2_x0[1] - cd.X0[1];
            cd.X0[2] = node2_x0[2] - cd.X0[2];
            return;
        }

        node_found = false;
        for (int i = 0; i < beam_node_state_indices.extent_int(0); i++) {
            if (cd.base_node_index == beam_node_state_indices[i]) {
                node1_x0[0] = beam_node_x0(i, 0);
                node1_x0[1] = beam_node_x0(i, 1);
                node1_x0[2] = beam_node_x0(i, 2);
                node_found = true;
                break;
            }
        }
        if (!node_found) {
            for (int i = 0; i < mass_node_state_indices.extent_int(0); i++) {
                if (cd.base_node_index == mass_node_state_indices[i]) {
                    node1_x0[0] = mass_node_x0(i, 0);
                    node1_x0[1] = mass_node_x0(i, 1);
                    node1_x0[2] = mass_node_x0(i, 2);
                    break;
                }
            }
        }

        cd.X0[0] = node2_x0[0] - node1_x0[0];
        cd.X0[1] = node2_x0[1] - node1_x0[1];
        cd.X0[2] = node2_x0[2] - node1_x0[2];
    }
};

}  // namespace openturbine
