#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

struct CopyConstraintsResidualToVector {
    Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type row_range;
    Kokkos::View<double*> residual;
    Kokkos::View<double*[6]>::const_type constraint_residual_terms;

    KOKKOS_FUNCTION
    void operator()(size_t i_constraint) const {
        const auto first_row = row_range(i_constraint).first;
        const auto num_rows = row_range(i_constraint).second - row_range(i_constraint).first;
        for(auto i = 0U; i < num_rows; ++i) {
            residual(first_row + i) = constraint_residual_terms(i_constraint, i);
        }
    }
};
}
