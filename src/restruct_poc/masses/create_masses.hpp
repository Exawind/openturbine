#pragma once

#include "masses.hpp"
#include "masses_input.hpp"
#include "masses_populate_element_views.hpp"
#include "masses_set_node_state_indices.hpp"

namespace openturbine {

inline Masses CreateMasses(const MassesInput& masses_input, int first_node = 0) {
    // Initialize mass structure
    Masses masses(masses_input.NumNodes());

    // Create host mirrors for views to be initialized
    auto host_gravity = Kokkos::create_mirror(masses.gravity);
    auto host_node_x0 = Kokkos::create_mirror(masses.node_x0);
    auto host_node_Mstar = Kokkos::create_mirror(masses.node_Mstar);

    // Set gravity
    host_gravity(0) = masses_input.gravity[0];
    host_gravity(1) = masses_input.gravity[1];
    host_gravity(2) = masses_input.gravity[2];

    // Loop through elements and populate views
    for (size_t i = 0; i < masses_input.elements.size(); i++) {
        PopulateElementViews(
            masses_input.elements[i], Kokkos::subview(host_node_x0, i, Kokkos::ALL),
            Kokkos::subview(host_node_Mstar, i, Kokkos::ALL, Kokkos::ALL)
        );
    }

    // TODO: update for assembly where state may apply to multiple nodes in different elements
    Kokkos::parallel_for(
        "MassesSetNodeStateIndices", masses.num_nodes,
        MassesSetNodeStateIndices{masses.node_state_indices, first_node}
    );

    // Copy data from host to device
    Kokkos::deep_copy(masses.gravity, host_gravity);
    Kokkos::deep_copy(masses.node_x0, host_node_x0);
    Kokkos::deep_copy(masses.node_Mstar, host_node_Mstar);

    return masses;
}

}  // namespace openturbine
