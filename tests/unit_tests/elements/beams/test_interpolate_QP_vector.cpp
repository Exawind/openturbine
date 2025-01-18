#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "test_interpolate_QP.hpp"

#include "elements/beams/interpolate_QP_vector.hpp"

namespace openturbine::tests {

inline auto create_node_u_dot_OneNode() {
    constexpr auto num_nodes = size_t{1U};
    constexpr auto num_entries = num_nodes * 6;
    auto node_u = Kokkos::View<double[1][num_nodes][6]>("node_u_dot");
    auto node_u_mirror = Kokkos::create_mirror(node_u);

    auto host_data = std::array<double, num_entries>{1., 2., 3., 4., 5., 6.};
    auto node_u_host = Kokkos::View<double[1][num_nodes][6], Kokkos::HostSpace>(host_data.data());
    Kokkos::deep_copy(node_u_mirror, node_u_host);
    Kokkos::deep_copy(node_u, node_u_mirror);
    return node_u;
}

TEST(InterpolateQPVectorTests, OneNodeOneQP) {
    constexpr auto num_qp = size_t{1U};
    constexpr auto num_nodes = size_t{1U};
    auto shape_interp = create_shape_interp_OneNodeOneQP();
    auto node_u_dot = create_node_u_dot_OneNode();
    auto qp_u_dot = Kokkos::View<double[1][num_qp][3]>("qp_u_dot");
    Kokkos::parallel_for(
        num_qp,
        InterpolateQPVector{
            0U, num_nodes, shape_interp,
            Kokkos::subview(node_u_dot, Kokkos::ALL, Kokkos::ALL, Kokkos::pair(0, 3)), qp_u_dot
        }
    );
    auto qp_u_dot_mirror = Kokkos::create_mirror(qp_u_dot);
    Kokkos::deep_copy(qp_u_dot_mirror, qp_u_dot);
    constexpr auto tolerance = 1.e-16;
    EXPECT_NEAR(qp_u_dot_mirror(0, 0, 0), 2., tolerance);
    EXPECT_NEAR(qp_u_dot_mirror(0, 0, 1), 4., tolerance);
    EXPECT_NEAR(qp_u_dot_mirror(0, 0, 2), 6., tolerance);
}

TEST(InterpolateQPVectorTests, OneNodeTwoQP) {
    constexpr auto num_qp = size_t{2U};
    constexpr auto num_nodes = size_t{1U};
    auto shape_interp = create_shape_interp_OneNodeTwoQP();
    auto node_u_dot = create_node_u_dot_OneNode();
    auto qp_u_dot = Kokkos::View<double[1][num_qp][3]>("qp_u_dot");
    Kokkos::parallel_for(
        num_qp,
        InterpolateQPVector{
            0, num_nodes, shape_interp,
            Kokkos::subview(node_u_dot, Kokkos::ALL, Kokkos::ALL, Kokkos::pair(0, 3)), qp_u_dot
        }
    );
    auto qp_u_dot_mirror = Kokkos::create_mirror(qp_u_dot);
    Kokkos::deep_copy(qp_u_dot_mirror, qp_u_dot);
    constexpr auto tolerance = 1.e-16;
    EXPECT_NEAR(qp_u_dot_mirror(0, 0, 0), 2., tolerance);
    EXPECT_NEAR(qp_u_dot_mirror(0, 0, 1), 4., tolerance);
    EXPECT_NEAR(qp_u_dot_mirror(0, 0, 2), 6., tolerance);

    EXPECT_NEAR(qp_u_dot_mirror(0, 1, 0), 4., tolerance);
    EXPECT_NEAR(qp_u_dot_mirror(0, 1, 1), 8., tolerance);
    EXPECT_NEAR(qp_u_dot_mirror(0, 1, 2), 12., tolerance);
}

inline auto create_node_u_dot_TwoNode() {
    constexpr auto num_nodes = 2;
    constexpr auto num_entries = num_nodes * 6;
    auto node_u_dot = Kokkos::View<double[1][num_nodes][6]>("node_u_dot");
    auto node_u_dot_mirror = Kokkos::create_mirror(node_u_dot);

    auto host_data =
        std::array<double, num_entries>{1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
    auto node_u_dot_host =
        Kokkos::View<double[1][num_nodes][6], Kokkos::HostSpace>(host_data.data());
    Kokkos::deep_copy(node_u_dot_mirror, node_u_dot_host);
    Kokkos::deep_copy(node_u_dot, node_u_dot_mirror);
    return node_u_dot;
}

TEST(InterpolateQPVectorTests, TwoNodeTwoQP) {
    constexpr auto num_qp = size_t{2U};
    constexpr auto num_nodes = size_t{2U};
    auto shape_interp = create_shape_interp_TwoNodeTwoQP();
    auto node_u_dot = create_node_u_dot_TwoNode();
    auto qp_u_dot = Kokkos::View<double[1][num_qp][3]>("qp_u_dot");
    Kokkos::parallel_for(
        num_qp,
        InterpolateQPVector{
            0U, num_nodes, shape_interp,
            Kokkos::subview(node_u_dot, Kokkos::ALL, Kokkos::ALL, Kokkos::pair(0, 3)), qp_u_dot
        }
    );
    auto qp_u_dot_mirror = Kokkos::create_mirror(qp_u_dot);
    Kokkos::deep_copy(qp_u_dot_mirror, qp_u_dot);
    constexpr auto tolerance = 1.e-16;
    EXPECT_NEAR(qp_u_dot_mirror(0, 0, 0), 30., tolerance);
    EXPECT_NEAR(qp_u_dot_mirror(0, 0, 1), 36., tolerance);
    EXPECT_NEAR(qp_u_dot_mirror(0, 0, 2), 42., tolerance);

    EXPECT_NEAR(qp_u_dot_mirror(0, 1, 0), 38., tolerance);
    EXPECT_NEAR(qp_u_dot_mirror(0, 1, 1), 46., tolerance);
    EXPECT_NEAR(qp_u_dot_mirror(0, 1, 2), 54., tolerance);
}

}  // namespace openturbine::tests
