#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename DeviceType>
struct InterpolateQPVector {
    size_t element;
    size_t num_nodes;
    typename Kokkos::View<double***, DeviceType>::const_type shape_interp;
    typename Kokkos::View<double** [3], Kokkos::LayoutStride, DeviceType>::const_type node_vector;
    Kokkos::View<double** [3], DeviceType> qp_vector;

    KOKKOS_FUNCTION
    void operator()(size_t qp) const {
        // const auto j = first_qp + qp;
        auto local_total = Kokkos::Array<double, 3>{};
        for (auto node = 0U; node < num_nodes; ++node) {
            const auto phi = shape_interp(element, node, qp);
            for (auto component = 0U; component < 3U; ++component) {
                local_total[component] += node_vector(element, node, component) * phi;
            }
        }
        for (auto component = 0U; component < 3U; ++component) {
            qp_vector(element, qp, component) = local_total[component];
        }
    }
};

}  // namespace openturbine
