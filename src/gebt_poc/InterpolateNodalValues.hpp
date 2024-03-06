#pragma once
#include <vector>

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/gebt_poc/types.hpp"

namespace openturbine::gebt_poc {

inline void InterpolateNodalValues(
    View2D::const_type nodal_values, std::vector<double> interpolation_function,
    View1D interpolated_values
) {
    Kokkos::deep_copy(interpolated_values, 0.);
    const auto n_nodes = nodal_values.extent(0);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        KokkosBlas::axpy(
            interpolation_function[i], Kokkos::subview(nodal_values, i, Kokkos::ALL),
            interpolated_values
        );
    }

    // Normalize the rotation quaternion if it is not already normalized
    if (nodal_values.extent(1) == LieGroupComponents) {
        auto q = Kokkos::subview(interpolated_values, Kokkos::pair(3, 7));
        if (auto norm = KokkosBlas::nrm2(q); norm != 0. && norm != 1.) {
            KokkosBlas::scal(q, 1. / norm, q);
        }
    }
}

}  // namespace openturbine::gebt_poc