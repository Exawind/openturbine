#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename DeviceType>
struct InterpolateQPVector {
    size_t i_elem;
    size_t num_nodes;
    typename Kokkos::View<double***, DeviceType>::const_type shape_interp;
    typename Kokkos::View<double** [3], Kokkos::LayoutStride, DeviceType>::const_type node_vector;
    Kokkos::View<double** [3], DeviceType> qp_vector;

    KOKKOS_FUNCTION
    void operator()(size_t j_index) const {
        // const auto j = first_qp + j_index;
        auto local_total = Kokkos::Array<double, 3>{};
        for (auto i_index = 0U; i_index < num_nodes; ++i_index) {
            const auto phi = shape_interp(i_elem, i_index, j_index);
            for (auto k = 0U; k < 3U; ++k) {
                local_total[k] += node_vector(i_elem, i_index, k) * phi;
            }
        }
        for (auto k = 0U; k < 3U; ++k) {
            qp_vector(i_elem, j_index, k) = local_total[k];
        }
    }
};

}  // namespace openturbine
