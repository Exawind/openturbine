#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

struct ContributeConstraintsSystemResidualToVector {
    Kokkos::View<size_t*>::const_type target_node_index;
    Kokkos::View<double*> residual;
    Kokkos::View<double* [6]>::const_type system_residual_terms;

    KOKKOS_FUNCTION
    void operator()(size_t i_constraint) const {
        const auto node_index = target_node_index(i_constraint);
        for (auto i = 0U; i < 6U; ++i) {
            residual((node_index * 6U) + i) += system_residual_terms(i_constraint, i);
        }
    }
};

}  // namespace openturbine
