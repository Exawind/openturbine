#pragma once

#include <vector>

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/gebt_poc/types.hpp"

namespace openturbine::gebt_poc {

inline void InterpolateNodalValueDerivatives(
    View1D::const_type nodal_values, std::vector<double> interpolation_function, double jacobian,
    View1D interpolated_values
) {
    if (jacobian == 0.) {
        throw std::invalid_argument("jacobian must be nonzero");
    }
    const auto n_nodes = nodal_values.extent(0) / LieGroupComponents;
    KokkosBlas::fill(interpolated_values, 0.);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        auto index = i * LieGroupComponents;
        KokkosBlas::axpy(
            interpolation_function[i],
            Kokkos::subview(nodal_values, Kokkos::pair(index, index + LieGroupComponents)),
            interpolated_values
        );
    }
    KokkosBlas::scal(interpolated_values, 1. / jacobian, interpolated_values);
}

inline void InterpolateNodalValueDerivatives(
    View2D::const_type nodal_values, std::vector<double> interpolation_function, double jacobian,
    View1D interpolated_values
) {
    if (jacobian == 0.) {
        throw std::invalid_argument("jacobian must be nonzero");
    }
    const auto n_nodes = nodal_values.extent(0);
    KokkosBlas::fill(interpolated_values, 0.);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        KokkosBlas::axpy(
            interpolation_function[i], Kokkos::subview(nodal_values, i, Kokkos::ALL),
            interpolated_values
        );
    }
    KokkosBlas::scal(interpolated_values, 1. / jacobian, interpolated_values);
}

}  // namespace openturbine::gebt_poc