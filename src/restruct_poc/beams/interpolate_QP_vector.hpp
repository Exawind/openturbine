#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct InterpolateQPVector {
    size_t i_elem;
    size_t first_qp;
    size_t first_node;
    size_t num_nodes;
    Kokkos::View<double***>::const_type shape_interp;
    View_Nx3::const_type node_vector;
    View_Nx3 qp_vector;

    KOKKOS_FUNCTION
    void operator()(size_t j_index) const {
        const auto j = first_qp + j_index;
        auto local_total = Kokkos::Array<double, 3>{};
        for (auto i_index = 0U; i_index < num_nodes; ++i_index) {
            const auto i = first_node + i_index;
            const auto phi = shape_interp(i_elem, i_index, j_index);
            for (auto k = 0U; k < 3U; ++k) {
                local_total[k] += node_vector(i, k) * phi;
            }
        }
        for (auto k = 0U; k < 3U; ++k) {
            qp_vector(j, k) = local_total[k];
        }
    }
};

}  // namespace openturbine