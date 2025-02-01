#pragma once

#include <Kokkos_Core.hpp>

#include "system/springs/calculate_distance_components.hpp"
#include "system/springs/calculate_force_coefficients.hpp"
#include "system/springs/calculate_force_vectors.hpp"
#include "system/springs/calculate_length.hpp"
#include "system/springs/calculate_stiffness_matrix.hpp"

namespace openturbine::springs {

struct CalculateQuadraturePointValues {
    Kokkos::View<double* [3]>::const_type x0;
    Kokkos::View<double* [3]>::const_type u1;
    Kokkos::View<double* [3]>::const_type u2;
    Kokkos::View<double*>::const_type l_ref;
    Kokkos::View<double*>::const_type k;

    Kokkos::View<double* [3]> r;
    Kokkos::View<double*> l;
    Kokkos::View<double*> c1;
    Kokkos::View<double*> c2;
    Kokkos::View<double* [3]> f;
    Kokkos::View<double* [3][3]> r_tilde;
    Kokkos::View<double* [3][3]> a;

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        // Calculate the relative distance vector between the two nodes
        springs::CalculateDistanceComponents{i_elem, x0, u1, u2, r}();

        // Calculate the current length of the spring
        springs::CalculateLength{i_elem, r, l}();

        // Calculate the force coefficients
        springs::CalculateForceCoefficients{i_elem, k, l_ref, l, c1, c2}();

        // Calculate the force vector
        springs::CalculateForceVectors{i_elem, r, c1, f}();

        // Calculate the stiffness matrix
        springs::CalculateStiffnessMatrix{i_elem, c1, c2, r, l, r_tilde, a}();
    }
};

}  // namespace openturbine::springs
