#pragma once

#include <numeric>

#include <KokkosBlas.hpp>

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct Masses {
    int num_nodes;  // Number of elements/nodes

    Kokkos::View<int*> node_state_indices;  // State row index for each node

    View_3 gravity;

    View_Nx7 node_x0;       // Inital position/rotation
    View_Nx7 node_u;        // State: translation/rotation displacement
    View_Nx6 node_u_dot;    // State: translation/rotation velocity
    View_Nx6 node_u_ddot;   // State: translation/rotation acceleration
    View_Nx6x6 node_Muu;    // Mass in global frame
    View_Nx6x6 node_Mstar;  // Mass matrix in material frame
    View_Nx6x6 node_R0;     // Initial rotation
    View_Nx6x6 node_RR0;    // Global rotation
    View_Nx6 node_FG;       // Gravity forces
    View_Nx6 node_FI;       // Inertial forces

    Masses() {}  // Default constructor which doesn't initialize views

    // Constructor which initializes views based on given sizes
    Masses(const int n_nodes)
        : num_nodes(n_nodes),
          // Element Data
          node_state_indices("node_state_indices", n_nodes),
          gravity("gravity"),
          // Node Data
          node_x0("node_x0", n_nodes),
          node_u("node_u", n_nodes),
          node_u_dot("node_u_dot", n_nodes),
          node_u_ddot("node_u_ddot", n_nodes),
          node_Muu("node_Muu", n_nodes),
          node_Mstar("node_Mstar", n_nodes),
          node_RR0("node_RR0", n_nodes),
          node_FG("node_FG", n_nodes),
          node_FI("node_FI", n_nodes) {}
};

}  // namespace openturbine
