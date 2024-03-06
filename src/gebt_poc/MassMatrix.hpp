#pragma once
#include <Kokkos_Core.hpp>

#include "src/gebt_poc/types.hpp"

namespace openturbine::gebt_poc {
inline double GetMass(View2D_6x6::const_type mass_matrix) {
    double mass;
    Kokkos::deep_copy(mass, Kokkos::subview(mass_matrix, 0, 0));
    return mass;
}

inline View1D_Vector GetCenterOfMass(View2D_6x6::const_type mass_matrix) {
    auto center_of_mass = View1D_Vector("center of mass");
    Kokkos::parallel_for(1, KOKKOS_LAMBDA(std::size_t) {
        double scale = 2. * mass_matrix(0, 0);
        center_of_mass(0) = (mass_matrix(5, 1) - mass_matrix(4, 2)) / scale;
        center_of_mass(1) = (mass_matrix(3, 2) - mass_matrix(5, 0)) / scale;
        center_of_mass(2) = (mass_matrix(4, 0) - mass_matrix(3, 1)) / scale;
    });
    return center_of_mass;
}

inline View2D_3x3 GetMomentOfInertia(View2D_6x6::const_type mass_matrix) {
    auto moment_of_inertia = View2D_3x3("moment of inertia");
    Kokkos::deep_copy(moment_of_inertia, Kokkos::subview(mass_matrix, Kokkos::make_pair(3, 6), Kokkos::make_pair(3, 6)));
    return moment_of_inertia;
}
}