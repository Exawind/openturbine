#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "test_interpolate_QP.hpp"

#include "src/restruct_poc/beams/interpolate_QP_state.hpp"

namespace openturbine::restruct_poc::tests {

inline auto create_node_u_OneNode() {
    constexpr auto num_nodes = 1;
    constexpr auto num_entries = num_nodes * 7;
    auto node_u = Kokkos::View<double[num_nodes][7]>("node_u");
    auto node_u_mirror = Kokkos::create_mirror(node_u);

    auto host_data = std::array<double, num_entries>{1., 2., 3., 4., 5., 6., 7.};
    auto node_u_host = Kokkos::View<double[num_nodes][7], Kokkos::HostSpace>(host_data.data());
    Kokkos::deep_copy(node_u_mirror, node_u_host);
    Kokkos::deep_copy(node_u, node_u_mirror);
    return node_u;
}

TEST(InterpolateQPStateTests, u_OneNodeOneQP) {
    constexpr auto first_qp = 0;
    constexpr auto num_qp = 1;
    constexpr auto first_node = 0;
    constexpr auto num_nodes = 1;
    auto shape_interp = create_shape_interp_OneNodeOneQP();
    auto node_u = create_node_u_OneNode();
    auto qp_u = Kokkos::View<double[num_qp][3]>("qp_u");
    Kokkos::parallel_for(num_qp, InterpolateQPState_u{first_qp, first_node, num_nodes, shape_interp, node_u, qp_u});
    auto qp_u_mirror = Kokkos::create_mirror(qp_u);
    Kokkos::deep_copy(qp_u_mirror, qp_u);
    constexpr auto tolerance = 1.e-16;
    EXPECT_NEAR(qp_u_mirror(0, 0), 2., tolerance);
    EXPECT_NEAR(qp_u_mirror(0, 1), 4., tolerance);
    EXPECT_NEAR(qp_u_mirror(0, 2), 6., tolerance);
}

TEST(InterpolateQPStateTests, uprime_OneNodeOneQP) {
    constexpr auto first_qp = 0;
    constexpr auto num_qp = 1;
    constexpr auto first_node = 0;
    constexpr auto num_nodes = 1;
    auto shape_deriv = create_shape_deriv_OneNodeOneQP();
    auto jacobian = create_jacobian_OneQP();
    auto node_u = create_node_u_OneNode();
    auto qp_uprime = Kokkos::View<double[num_qp][3]>("qp_uprime");
    Kokkos::parallel_for(num_qp, InterpolateQPState_uprime{first_qp, first_node, num_nodes, shape_deriv, jacobian, node_u, qp_uprime});
    auto qp_uprime_mirror = Kokkos::create_mirror(qp_uprime);
    Kokkos::deep_copy(qp_uprime_mirror, qp_uprime);
    constexpr auto tolerance = 1.e-16;
    EXPECT_NEAR(qp_uprime_mirror(0, 0), 2., tolerance);
    EXPECT_NEAR(qp_uprime_mirror(0, 1), 4., tolerance);
    EXPECT_NEAR(qp_uprime_mirror(0, 2), 6., tolerance);
}

TEST(InterpolateQPStateTests, r_OneNodeOneQP) {
    constexpr auto first_qp = 0;
    constexpr auto num_qp = 1;
    constexpr auto first_node = 0;
    constexpr auto num_nodes = 1;
    auto shape_interp = create_shape_interp_OneNodeOneQP();
    auto node_u = create_node_u_OneNode();
    auto qp_r = Kokkos::View<double[num_qp][4]>("qp_r");
    Kokkos::parallel_for(num_qp, InterpolateQPState_r{first_qp, first_node, num_nodes, shape_interp, node_u, qp_r});
    auto qp_r_mirror = Kokkos::create_mirror(qp_r);
    Kokkos::deep_copy(qp_r_mirror, qp_r);
    constexpr auto tolerance = 1.e-16;
    EXPECT_NEAR(qp_r_mirror(0, 0), 8., tolerance);
    EXPECT_NEAR(qp_r_mirror(0, 1), 10., tolerance);
    EXPECT_NEAR(qp_r_mirror(0, 2), 12., tolerance);
    EXPECT_NEAR(qp_r_mirror(0, 3), 14., tolerance);
}

TEST(InterpolateQPStateTests, rprime_OneNodeOneQP) {
    constexpr auto first_qp = 0;
    constexpr auto num_qp = 1;
    constexpr auto first_node = 0;
    constexpr auto num_nodes = 1;
    auto shape_deriv = create_shape_deriv_OneNodeOneQP();
    auto jacobian = create_jacobian_OneQP();
    auto node_u = create_node_u_OneNode();
    auto qp_rprime = Kokkos::View<double[num_qp][4]>("qp_uprime");
    Kokkos::parallel_for(num_qp, InterpolateQPState_rprime{first_qp, first_node, num_nodes, shape_deriv, jacobian, node_u, qp_rprime});
    auto qp_rprime_mirror = Kokkos::create_mirror(qp_rprime);
    Kokkos::deep_copy(qp_rprime_mirror, qp_rprime);
    constexpr auto tolerance = 1.e-16;
    EXPECT_NEAR(qp_rprime_mirror(0, 0), 8., tolerance);
    EXPECT_NEAR(qp_rprime_mirror(0, 1), 10., tolerance);
    EXPECT_NEAR(qp_rprime_mirror(0, 2), 12., tolerance);
    EXPECT_NEAR(qp_rprime_mirror(0, 3), 14., tolerance);
}

TEST(InterpolateQPStateTests, u_OneNodeTwoQP) {
    constexpr auto first_qp = 0;
    constexpr auto num_qp = 2;
    constexpr auto first_node = 0;
    constexpr auto num_nodes = 1;
    auto shape_interp = create_shape_interp_OneNodeTwoQP();
    auto node_u = create_node_u_OneNode();
    auto qp_u = Kokkos::View<double[num_qp][3]>("qp_u");
    Kokkos::parallel_for(num_qp, InterpolateQPState_u{first_qp, first_node, num_nodes, shape_interp, node_u, qp_u});
    auto qp_u_mirror = Kokkos::create_mirror(qp_u);
    Kokkos::deep_copy(qp_u_mirror, qp_u);
    constexpr auto tolerance = 1.e-16;
    EXPECT_NEAR(qp_u_mirror(0, 0), 2., tolerance);
    EXPECT_NEAR(qp_u_mirror(0, 1), 4., tolerance);
    EXPECT_NEAR(qp_u_mirror(0, 2), 6., tolerance);

    EXPECT_NEAR(qp_u_mirror(1, 0), 4., tolerance);
    EXPECT_NEAR(qp_u_mirror(1, 1), 8., tolerance);
    EXPECT_NEAR(qp_u_mirror(1, 2), 12., tolerance);
}

TEST(InterpolateQPStateTests, uprime_OneNodeTwoQP) {
    constexpr auto first_qp = 0;
    constexpr auto num_qp = 2;
    constexpr auto first_node = 0;
    constexpr auto num_nodes = 1;
    auto shape_deriv = create_shape_deriv_OneNodeTwoQP();
    auto jacobian = create_jacobian_TwoQP();
    auto node_u = create_node_u_OneNode();
    auto qp_uprime = Kokkos::View<double[num_qp][3]>("qp_uprime");
    Kokkos::parallel_for(num_qp, InterpolateQPState_uprime{first_qp, first_node, num_nodes, shape_deriv, jacobian, node_u, qp_uprime});
    auto qp_uprime_mirror = Kokkos::create_mirror(qp_uprime);
    Kokkos::deep_copy(qp_uprime_mirror, qp_uprime);
    constexpr auto tolerance = 1.e-16;
    EXPECT_NEAR(qp_uprime_mirror(0, 0), 2., tolerance);
    EXPECT_NEAR(qp_uprime_mirror(0, 1), 4., tolerance);
    EXPECT_NEAR(qp_uprime_mirror(0, 2), 6., tolerance);

    EXPECT_NEAR(qp_uprime_mirror(1, 0), 3., tolerance);
    EXPECT_NEAR(qp_uprime_mirror(1, 1), 6., tolerance);
    EXPECT_NEAR(qp_uprime_mirror(1, 2), 9., tolerance);
}

TEST(InterpolateQPStateTests, r_OneNodeTwoQP) {
    constexpr auto first_qp = 0;
    constexpr auto num_qp = 2;
    constexpr auto first_node = 0;
    constexpr auto num_nodes = 1;
    auto shape_interp = create_shape_interp_OneNodeTwoQP();
    auto node_u = create_node_u_OneNode();
    auto qp_r = Kokkos::View<double[num_qp][4]>("qp_r");
    Kokkos::parallel_for(num_qp, InterpolateQPState_r{first_qp, first_node, num_nodes, shape_interp, node_u, qp_r});
    auto qp_r_mirror = Kokkos::create_mirror(qp_r);
    Kokkos::deep_copy(qp_r_mirror, qp_r);
    constexpr auto tolerance = 1.e-16;
    EXPECT_NEAR(qp_r_mirror(0, 0), 8., tolerance);
    EXPECT_NEAR(qp_r_mirror(0, 1), 10., tolerance);
    EXPECT_NEAR(qp_r_mirror(0, 2), 12., tolerance);
    EXPECT_NEAR(qp_r_mirror(0, 3), 14., tolerance);

    EXPECT_NEAR(qp_r_mirror(1, 0), 16., tolerance);
    EXPECT_NEAR(qp_r_mirror(1, 1), 20., tolerance);
    EXPECT_NEAR(qp_r_mirror(1, 2), 24., tolerance);
    EXPECT_NEAR(qp_r_mirror(1, 3), 28., tolerance);
}

TEST(InterpolateQPStateTests, rprime_OneNodeTwoQP) {
    constexpr auto first_qp = 0;
    constexpr auto num_qp = 2;
    constexpr auto first_node = 0;
    constexpr auto num_nodes = 1;
    auto shape_deriv = create_shape_deriv_OneNodeTwoQP();
    auto jacobian = create_jacobian_TwoQP();
    auto node_u = create_node_u_OneNode();
    auto qp_rprime = Kokkos::View<double[num_qp][4]>("qp_uprime");
    Kokkos::parallel_for(num_qp, InterpolateQPState_rprime{first_qp, first_node, num_nodes, shape_deriv, jacobian, node_u, qp_rprime});
    auto qp_rprime_mirror = Kokkos::create_mirror(qp_rprime);
    Kokkos::deep_copy(qp_rprime_mirror, qp_rprime);
    constexpr auto tolerance = 1.e-16;
    EXPECT_NEAR(qp_rprime_mirror(0, 0), 8., tolerance);
    EXPECT_NEAR(qp_rprime_mirror(0, 1), 10., tolerance);
    EXPECT_NEAR(qp_rprime_mirror(0, 2), 12., tolerance);
    EXPECT_NEAR(qp_rprime_mirror(0, 3), 14., tolerance);

    EXPECT_NEAR(qp_rprime_mirror(1, 0), 12., tolerance);
    EXPECT_NEAR(qp_rprime_mirror(1, 1), 15., tolerance);
    EXPECT_NEAR(qp_rprime_mirror(1, 2), 18., tolerance);
    EXPECT_NEAR(qp_rprime_mirror(1, 3), 21., tolerance);
}

inline auto create_node_u_TwoNode() {
    constexpr auto num_nodes = 2;
    constexpr auto num_entries = num_nodes * 7;
    auto node_u = Kokkos::View<double[num_nodes][7]>("node_u");
    auto node_u_mirror = Kokkos::create_mirror(node_u);

    auto host_data = std::array<double, num_entries>{1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14.};
    auto node_u_host = Kokkos::View<double[num_nodes][7], Kokkos::HostSpace>(host_data.data());
    Kokkos::deep_copy(node_u_mirror, node_u_host);
    Kokkos::deep_copy(node_u, node_u_mirror);
    return node_u;
}

TEST(InterpolateQPStateTests, u_TwoNodeTwoQP) {
    constexpr auto first_qp = 0;
    constexpr auto num_qp = 2;
    constexpr auto first_node = 0;
    constexpr auto num_nodes = 2;
    auto shape_interp = create_shape_interp_TwoNodeTwoQP();
    auto node_u = create_node_u_TwoNode();
    auto qp_u = Kokkos::View<double[num_qp][3]>("qp_u");
    Kokkos::parallel_for(num_qp, InterpolateQPState_u{first_qp, first_node, num_nodes, shape_interp, node_u, qp_u});
    auto qp_u_mirror = Kokkos::create_mirror(qp_u);
    Kokkos::deep_copy(qp_u_mirror, qp_u);
    constexpr auto tolerance = 1.e-16;
    EXPECT_NEAR(qp_u_mirror(0, 0), 34., tolerance);
    EXPECT_NEAR(qp_u_mirror(0, 1), 40., tolerance);
    EXPECT_NEAR(qp_u_mirror(0, 2), 46., tolerance);

    EXPECT_NEAR(qp_u_mirror(1, 0), 43., tolerance);
    EXPECT_NEAR(qp_u_mirror(1, 1), 51., tolerance);
    EXPECT_NEAR(qp_u_mirror(1, 2), 59., tolerance);
}

TEST(InterpolateQPStateTests, uprime_TwoNodeTwoQP) {
    constexpr auto first_qp = 0;
    constexpr auto num_qp = 2;
    constexpr auto first_node = 0;
    constexpr auto num_nodes = 2;
    auto shape_deriv = create_shape_deriv_TwoNodeTwoQP();
    auto jacobian = create_jacobian_TwoQP();
    auto node_u = create_node_u_TwoNode();
    auto qp_uprime = Kokkos::View<double[num_qp][3]>("qp_uprime");
    Kokkos::parallel_for(num_qp, InterpolateQPState_uprime{first_qp, first_node, num_nodes, shape_deriv, jacobian, node_u, qp_uprime});
    auto qp_uprime_mirror = Kokkos::create_mirror(qp_uprime);
    Kokkos::deep_copy(qp_uprime_mirror, qp_uprime);
    constexpr auto tolerance = 1.e-16;
    EXPECT_NEAR(qp_uprime_mirror(0, 0), 34., tolerance);
    EXPECT_NEAR(qp_uprime_mirror(0, 1), 40., tolerance);
    EXPECT_NEAR(qp_uprime_mirror(0, 2), 46., tolerance);

    EXPECT_NEAR(qp_uprime_mirror(1, 0), 43., tolerance);
    EXPECT_NEAR(qp_uprime_mirror(1, 1), 51., tolerance);
    EXPECT_NEAR(qp_uprime_mirror(1, 2), 59., tolerance);
}

TEST(InterpolateQPStateTests, r_TwoNodeTwoQP) {
    constexpr auto first_qp = 0;
    constexpr auto num_qp = 2;
    constexpr auto first_node = 0;
    constexpr auto num_nodes = 2;
    auto shape_interp = create_shape_interp_TwoNodeTwoQP();
    auto node_u = create_node_u_TwoNode();
    auto qp_r = Kokkos::View<double[num_qp][4]>("qp_r");
    Kokkos::parallel_for(num_qp, InterpolateQPState_r{first_qp, first_node, num_nodes, shape_interp, node_u, qp_r});
    auto qp_r_mirror = Kokkos::create_mirror(qp_r);
    Kokkos::deep_copy(qp_r_mirror, qp_r);
    constexpr auto tolerance = 1.e-16;
    EXPECT_NEAR(qp_r_mirror(0, 0), 52., tolerance);
    EXPECT_NEAR(qp_r_mirror(0, 1), 58., tolerance);
    EXPECT_NEAR(qp_r_mirror(0, 2), 64., tolerance);
    EXPECT_NEAR(qp_r_mirror(0, 3), 70., tolerance);

    EXPECT_NEAR(qp_r_mirror(1, 0), 67., tolerance);
    EXPECT_NEAR(qp_r_mirror(1, 1), 75., tolerance);
    EXPECT_NEAR(qp_r_mirror(1, 2), 83., tolerance);
    EXPECT_NEAR(qp_r_mirror(1, 3), 91., tolerance);
}

TEST(InterpolateQPStateTests, rprime_TwoNodeTwoQP) {
    constexpr auto first_qp = 0;
    constexpr auto num_qp = 2;
    constexpr auto first_node = 0;
    constexpr auto num_nodes = 2;
    auto shape_deriv = create_shape_deriv_TwoNodeTwoQP();
    auto jacobian = create_jacobian_TwoQP();
    auto node_u = create_node_u_TwoNode();
    auto qp_rprime = Kokkos::View<double[num_qp][4]>("qp_uprime");
    Kokkos::parallel_for(num_qp, InterpolateQPState_rprime{first_qp, first_node, num_nodes, shape_deriv, jacobian, node_u, qp_rprime});
    auto qp_rprime_mirror = Kokkos::create_mirror(qp_rprime);
    Kokkos::deep_copy(qp_rprime_mirror, qp_rprime);
    constexpr auto tolerance = 1.e-16;
    EXPECT_NEAR(qp_rprime_mirror(0, 0), 52., tolerance);
    EXPECT_NEAR(qp_rprime_mirror(0, 1), 58., tolerance);
    EXPECT_NEAR(qp_rprime_mirror(0, 2), 64., tolerance);
    EXPECT_NEAR(qp_rprime_mirror(0, 3), 70., tolerance);

    EXPECT_NEAR(qp_rprime_mirror(1, 0), 67., tolerance);
    EXPECT_NEAR(qp_rprime_mirror(1, 1), 75., tolerance);
    EXPECT_NEAR(qp_rprime_mirror(1, 2), 83., tolerance);
    EXPECT_NEAR(qp_rprime_mirror(1, 3), 91., tolerance);
}

}