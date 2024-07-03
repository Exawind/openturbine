#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "test_interpolate_QP.hpp"

#include "src/restruct_poc/beams/interpolate_QP_velocity.hpp"

namespace openturbine::restruct_poc::tests {

inline auto create_node_u_dot_OneNode() {
    constexpr auto num_nodes = 1;
    constexpr auto num_entries = num_nodes * 6;
    auto node_u = Kokkos::View<double[num_nodes][6]>("node_u_dot");
    auto node_u_mirror = Kokkos::create_mirror(node_u);

    auto host_data = std::array<double, num_entries>{1., 2., 3., 4., 5., 6.};
    auto node_u_host = Kokkos::View<double[num_nodes][6], Kokkos::HostSpace>(host_data.data());
    Kokkos::deep_copy(node_u_mirror, node_u_host);
    Kokkos::deep_copy(node_u, node_u_mirror);
    return node_u;
}

TEST(InterpolateQPVelocityTests, Translational_OneNodeOneQP) {
    constexpr auto first_qp = 0;
    constexpr auto num_qp = 1;
    constexpr auto first_node = 0;
    constexpr auto num_nodes = 1;
    auto shape_interp = create_shape_interp_OneNodeOneQP();
    auto node_u_dot = create_node_u_dot_OneNode();
    auto qp_u_dot = Kokkos::View<double[num_qp][3]>("qp_u_dot");
    Kokkos::parallel_for(num_qp, InterpolateQPVelocity_Translational{first_qp, first_node, num_nodes, shape_interp, node_u_dot, qp_u_dot});
    auto qp_u_dot_mirror = Kokkos::create_mirror(qp_u_dot);
    Kokkos::deep_copy(qp_u_dot_mirror, qp_u_dot);
    constexpr auto tolerance = 1.e-16;
    EXPECT_NEAR(qp_u_dot_mirror(0, 0), 2., tolerance);
    EXPECT_NEAR(qp_u_dot_mirror(0, 1), 4., tolerance);
    EXPECT_NEAR(qp_u_dot_mirror(0, 2), 6., tolerance);
}

TEST(InterpolateQPVelocityTests, Angular_OneNodeOneQP) {
    constexpr auto first_qp = 0;
    constexpr auto num_qp = 1;
    constexpr auto first_node = 0;
    constexpr auto num_nodes = 1;
    auto shape_interp = create_shape_interp_OneNodeOneQP();
    auto node_u_dot = create_node_u_dot_OneNode();
    auto qp_u_omega = Kokkos::View<double[num_qp][3]>("qp_u_dot");
    Kokkos::parallel_for(num_qp, InterpolateQPVelocity_Angular{first_qp, first_node, num_nodes, shape_interp, node_u_dot, qp_u_omega});
    auto qp_u_omega_mirror = Kokkos::create_mirror(qp_u_omega);
    Kokkos::deep_copy(qp_u_omega_mirror, qp_u_omega);
    constexpr auto tolerance = 1.e-16;
    EXPECT_NEAR(qp_u_omega_mirror(0, 0), 8., tolerance);
    EXPECT_NEAR(qp_u_omega_mirror(0, 1), 10., tolerance);
    EXPECT_NEAR(qp_u_omega_mirror(0, 2), 12., tolerance);
}

TEST(InterpolateQPVelocityTests, Translational_OneNodeTwoQP) {
    constexpr auto first_qp = 0;
    constexpr auto num_qp = 2;
    constexpr auto first_node = 0;
    constexpr auto num_nodes = 1;
    auto shape_interp = create_shape_interp_OneNodeTwoQP();
    auto node_u_dot = create_node_u_dot_OneNode();
    auto qp_u_dot = Kokkos::View<double[num_qp][3]>("qp_u_dot");
    Kokkos::parallel_for(num_qp, InterpolateQPVelocity_Translational{first_qp, first_node, num_nodes, shape_interp, node_u_dot, qp_u_dot});
    auto qp_u_dot_mirror = Kokkos::create_mirror(qp_u_dot);
    Kokkos::deep_copy(qp_u_dot_mirror, qp_u_dot);
    constexpr auto tolerance = 1.e-16;
    EXPECT_NEAR(qp_u_dot_mirror(0, 0), 2., tolerance);
    EXPECT_NEAR(qp_u_dot_mirror(0, 1), 4., tolerance);
    EXPECT_NEAR(qp_u_dot_mirror(0, 2), 6., tolerance);

    EXPECT_NEAR(qp_u_dot_mirror(1, 0), 4., tolerance);
    EXPECT_NEAR(qp_u_dot_mirror(1, 1), 8., tolerance);
    EXPECT_NEAR(qp_u_dot_mirror(1, 2), 12., tolerance);
}

TEST(InterpolateQPVelocityTests, Angular_OneNodeTowQP) {
    constexpr auto first_qp = 0;
    constexpr auto num_qp = 2;
    constexpr auto first_node = 0;
    constexpr auto num_nodes = 1;
    auto shape_interp = create_shape_interp_OneNodeTwoQP();
    auto node_u_dot = create_node_u_dot_OneNode();
    auto qp_u_omega = Kokkos::View<double[num_qp][3]>("qp_u_dot");
    Kokkos::parallel_for(num_qp, InterpolateQPVelocity_Angular{first_qp, first_node, num_nodes, shape_interp, node_u_dot, qp_u_omega});
    auto qp_u_omega_mirror = Kokkos::create_mirror(qp_u_omega);
    Kokkos::deep_copy(qp_u_omega_mirror, qp_u_omega);
    constexpr auto tolerance = 1.e-16;
    EXPECT_NEAR(qp_u_omega_mirror(0, 0), 8., tolerance);
    EXPECT_NEAR(qp_u_omega_mirror(0, 1), 10., tolerance);
    EXPECT_NEAR(qp_u_omega_mirror(0, 2), 12., tolerance);

    EXPECT_NEAR(qp_u_omega_mirror(1, 0), 16., tolerance);
    EXPECT_NEAR(qp_u_omega_mirror(1, 1), 20., tolerance);
    EXPECT_NEAR(qp_u_omega_mirror(1, 2), 24., tolerance);
}

inline auto create_node_u_dot_TwoNode() {
    constexpr auto num_nodes = 2;
    constexpr auto num_entries = num_nodes * 6;
    auto node_u_dot = Kokkos::View<double[num_nodes][6]>("node_u_dot");
    auto node_u_dot_mirror = Kokkos::create_mirror(node_u_dot);

    auto host_data = std::array<double, num_entries>{1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
    auto node_u_dot_host = Kokkos::View<double[num_nodes][6], Kokkos::HostSpace>(host_data.data());
    Kokkos::deep_copy(node_u_dot_mirror, node_u_dot_host);
    Kokkos::deep_copy(node_u_dot, node_u_dot_mirror);
    return node_u_dot;
}

TEST(InterpolateQPVelocityTests, Translational_TwoNodeTwoQP) {
    constexpr auto first_qp = 0;
    constexpr auto num_qp = 2;
    constexpr auto first_node = 0;
    constexpr auto num_nodes = 2;
    auto shape_interp = create_shape_interp_TwoNodeTwoQP();
    auto node_u_dot = create_node_u_dot_TwoNode();
    auto qp_u_dot = Kokkos::View<double[num_qp][3]>("qp_u_dot");
    Kokkos::parallel_for(num_qp, InterpolateQPVelocity_Translational{first_qp, first_node, num_nodes, shape_interp, node_u_dot, qp_u_dot});
    auto qp_u_dot_mirror = Kokkos::create_mirror(qp_u_dot);
    Kokkos::deep_copy(qp_u_dot_mirror, qp_u_dot);
    constexpr auto tolerance = 1.e-16;
    EXPECT_NEAR(qp_u_dot_mirror(0, 0), 30., tolerance);
    EXPECT_NEAR(qp_u_dot_mirror(0, 1), 36., tolerance);
    EXPECT_NEAR(qp_u_dot_mirror(0, 2), 42., tolerance);

    EXPECT_NEAR(qp_u_dot_mirror(1, 0), 38., tolerance);
    EXPECT_NEAR(qp_u_dot_mirror(1, 1), 46., tolerance);
    EXPECT_NEAR(qp_u_dot_mirror(1, 2), 54., tolerance);
}

TEST(InterpolateQPVelocityTests, Angular_TwoNodeTowQP) {
    constexpr auto first_qp = 0;
    constexpr auto num_qp = 2;
    constexpr auto first_node = 0;
    constexpr auto num_nodes = 2;
    auto shape_interp = create_shape_interp_OneNodeTwoQP();
    auto node_u_dot = create_node_u_dot_TwoNode();
    auto qp_u_omega = Kokkos::View<double[num_qp][3]>("qp_u_dot");
    Kokkos::parallel_for(num_qp, InterpolateQPVelocity_Angular{first_qp, first_node, num_nodes, shape_interp, node_u_dot, qp_u_omega});
    auto qp_u_omega_mirror = Kokkos::create_mirror(qp_u_omega);
    Kokkos::deep_copy(qp_u_omega_mirror, qp_u_omega);
    constexpr auto tolerance = 1.e-16;
    EXPECT_NEAR(qp_u_omega_mirror(0, 0), 48., tolerance);
    EXPECT_NEAR(qp_u_omega_mirror(0, 1), 54., tolerance);
    EXPECT_NEAR(qp_u_omega_mirror(0, 2), 60., tolerance);

    EXPECT_NEAR(qp_u_omega_mirror(1, 0), 46., tolerance);
    EXPECT_NEAR(qp_u_omega_mirror(1, 1), 53., tolerance);
    EXPECT_NEAR(qp_u_omega_mirror(1, 2), 60., tolerance);
}

}