#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/solver/populate_tangent_indices.hpp"

namespace openturbine::tests {

TEST(PopulateTangentIndices, TwoNodes) {
    constexpr auto num_system_nodes = 2U;
    constexpr auto num_system_dofs = num_system_nodes * 6U * 6U;
    constexpr auto node_state_indices_host_data = std::array{size_t{0U}, size_t{1U}};
    const auto node_state_indices_host =
        Kokkos::View<const size_t[num_system_nodes], Kokkos::HostSpace>(
            node_state_indices_host_data.data()
        );
    const auto node_state_indices = Kokkos::View<size_t[num_system_nodes]>("node_state_indices");
    Kokkos::deep_copy(node_state_indices, node_state_indices_host);

    const auto indices = Kokkos::View<int[num_system_dofs]>("indices");

    Kokkos::parallel_for(
        "PopulateTangentIndices", 1,
        PopulateTangentIndices{num_system_nodes, node_state_indices, indices}
    );

    constexpr auto indices_exact_data = std::array{
        0, 1, 2, 3, 4,  5,  0, 1, 2, 3, 4,  5,  0, 1, 2, 3, 4,  5,  0, 1, 2, 3, 4,  5,
        0, 1, 2, 3, 4,  5,  0, 1, 2, 3, 4,  5,  6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11,
        6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11,
    };
    const auto indices_exact =
        Kokkos::View<const int[num_system_dofs], Kokkos::HostSpace>(indices_exact_data.data());

    const auto indices_host = Kokkos::create_mirror(indices);
    Kokkos::deep_copy(indices_host, indices);

    for (auto i = 0U; i < num_system_dofs; ++i) {
        EXPECT_EQ(indices_host(i), indices_exact(i));
    }
}

}  // namespace openturbine::tests
