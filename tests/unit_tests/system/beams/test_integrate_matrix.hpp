#include <array>
#include <string_view>

#include <Kokkos_Core.hpp>

namespace openturbine::tests {

template <size_t n_elem, size_t n_nodes>
auto get_node_state_indices() {
    using IndicesView = Kokkos::View<size_t[n_elem][n_nodes]>;
    auto indices = IndicesView("node_state_indices");
    const auto host_indices = Kokkos::create_mirror_view(indices);
    for (auto i = 0U; i < n_elem; ++i) {
        for (auto j = 0U; j < n_nodes; ++j) {
            host_indices(i, j) = i * n_nodes + j;
        }
    }
    Kokkos::deep_copy(indices, host_indices);
    return indices;
}
}  // namespace openturbine::tests
