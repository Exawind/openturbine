#pragma once

#include <Kokkos_Core.hpp>

#include "src/gebt_poc/types.hpp"

namespace openturbine::gebt_poc {

inline void CalculateSectionalStrain(
    View1D::const_type pos_vector_derivatives_qp, View1D::const_type gen_coords_derivatives_qp,
    View1D::const_type curvature, View1D sectional_strain
) {
    Kokkos::deep_copy(sectional_strain, 0.);
    // Calculate the sectional strain based on Eq. (35) in the "SO(3)-based GEBT Beam" document
    // in theory guide
    auto sectional_strain_1 = Kokkos::subview(sectional_strain, Kokkos::make_pair(0, 3));
    Kokkos::deep_copy(
        sectional_strain_1, Kokkos::subview(pos_vector_derivatives_qp, Kokkos::make_pair(0, 3))
    );
    KokkosBlas::axpy(
        1., Kokkos::subview(gen_coords_derivatives_qp, Kokkos::make_pair(0, 3)), sectional_strain_1
    );
    auto sectional_strain_2 = Kokkos::subview(sectional_strain, Kokkos::make_pair(3, 6));
    Kokkos::deep_copy(sectional_strain_2, curvature);
}

}  // namespace openturbine::gebt_poc