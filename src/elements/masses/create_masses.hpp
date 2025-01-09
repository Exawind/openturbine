#pragma once

#include "masses.hpp"
#include "masses_input.hpp"

#include "src/elements/beams/calculate_QP_position.hpp"

namespace openturbine {

inline Masses CreateMasses(const MassesInput& masses_input, const std::vector<Node>& nodes) {
    Masses masses(masses_input.NumElements());

    auto host_gravity = Kokkos::create_mirror(masses.gravity);
    auto host_state_indices = Kokkos::create_mirror(masses.state_indices);
    auto host_x0 = Kokkos::create_mirror(masses.node_x0);
    auto host_Mstar = Kokkos::create_mirror(masses.qp_Mstar);

    host_gravity(0) = masses_input.gravity[0];
    host_gravity(1) = masses_input.gravity[1];
    host_gravity(2) = masses_input.gravity[2];

    // Populate element data - x0 and Mstar
    for (size_t i_elem = 0; i_elem < masses_input.NumElements(); i_elem++) {
        host_state_indices(i_elem) = static_cast<size_t>(masses_input.elements[i_elem].node_id);
        const auto& elem = masses_input.elements[i_elem];
        for (auto i_dof = 0U; i_dof < nodes[elem.node_id].x.size(); ++i_dof) {
            host_x0(i_elem, i_dof) = nodes[elem.node_id].x[i_dof];
        }
        for (auto m = 0U; m < 6U; ++m) {
            for (auto n = 0U; n < 6U; ++n) {
                host_Mstar(i_elem, m, n) = elem.M_star[m][n];
            }
        }
    }

    Kokkos::deep_copy(masses.gravity, host_gravity);
    Kokkos::deep_copy(masses.state_indices, host_state_indices);
    Kokkos::deep_copy(masses.node_x0, host_x0);
    Kokkos::deep_copy(masses.qp_Mstar, host_Mstar);

    return masses;
}

}  // namespace openturbine
