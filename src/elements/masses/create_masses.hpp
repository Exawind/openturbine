#pragma once

#include "masses.hpp"
#include "masses_input.hpp"

#include "src/elements/beams/calculate_QP_position.hpp"

namespace openturbine {

inline Masses CreateMasses(const MassesInput& masses_input) {
    Masses masses(masses_input.NumElements());

    auto host_gravity = Kokkos::create_mirror(masses.gravity);
    auto host_state_indices = Kokkos::create_mirror(masses.state_indices);
    auto host_x0 = Kokkos::create_mirror(masses.x0);
    auto host_u = Kokkos::create_mirror(masses.u);
    auto host_u_dot = Kokkos::create_mirror(masses.u_dot);
    auto host_u_ddot = Kokkos::create_mirror(masses.u_ddot);
    auto host_Mstar = Kokkos::create_mirror(masses.Mstar);

    host_gravity(0) = masses_input.gravity[0];
    host_gravity(1) = masses_input.gravity[1];
    host_gravity(2) = masses_input.gravity[2];

    auto populate_masses_element_views = [&](const MassElement& elem, auto x0, auto Mstar) {
        // Populate initial position and orientation of the node
        for (size_t i_dof = 0; i_dof < elem.node.x.size(); ++i_dof) {
            x0(i_dof) = elem.node.x[i_dof];
        }

        // Populate the mass matrix at material frame
        Kokkos::deep_copy(Mstar, 0.);
        for (size_t m = 0; m < 6; ++m) {
            for (size_t n = 0; n < 6; ++n) {
                Mstar(m, n) = elem.M_star[m][n];
            }
        }
    };

    // Populate element data - x0 and Mstar
    for (size_t i_elem = 0; i_elem < masses_input.NumElements(); i_elem++) {
        host_state_indices(i_elem, 0U) = static_cast<size_t>(masses_input.elements[i_elem].node.ID);
        populate_masses_element_views(
            masses_input.elements[i_elem], Kokkos::subview(host_x0, i_elem, 0U, Kokkos::ALL),
            Kokkos::subview(host_Mstar, i_elem, 0U, Kokkos::ALL, Kokkos::ALL)
        );
    }

    Kokkos::deep_copy(masses.gravity, host_gravity);
    Kokkos::deep_copy(masses.state_indices, host_state_indices);
    Kokkos::deep_copy(masses.x0, host_x0);
    Kokkos::deep_copy(masses.u, host_u);
    Kokkos::deep_copy(masses.u_dot, host_u_dot);
    Kokkos::deep_copy(masses.u_ddot, host_u_ddot);
    Kokkos::deep_copy(masses.Mstar, host_Mstar);

    return masses;
}

}  // namespace openturbine
