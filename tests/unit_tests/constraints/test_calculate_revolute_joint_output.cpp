#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/constraints/calculate_revolute_joint_output.hpp"

namespace openturbine::tests {

TEST(CalculateRevoluteJointOutputTests, OneConstraint) {
    const auto target_node_index = Kokkos::View<size_t[1]>("target_node_index");
    constexpr auto target_node_index_host_data = std::array<size_t, 1>{1ul};
    const auto target_node_index_host = Kokkos::View<size_t[1], Kokkos::HostSpace>::const_type(target_node_index_host_data.data());
    Kokkos::deep_copy(target_node_index, target_node_index_host);

    const auto axes = Kokkos::View<double[1][3][3]>("axes");
    constexpr auto axes_host_data = std::array{1., 2., 3., 4., 5., 6., 7., 8., 9.};
    const auto axes_host = Kokkos::View<double[1][3][3], Kokkos::HostSpace>::const_type(axes_host_data.data());
    const auto axes_mirror = Kokkos::create_mirror(axes);
    Kokkos::deep_copy(axes_mirror, axes_host);
    Kokkos::deep_copy(axes, axes_mirror);

    const auto node_x0 = Kokkos::View<double[2][7]>("node_x0");
    constexpr auto node_x0_host_data = std::array{0., 0., 0., 0., 0., 0., 0., 11., 12., 13., 14., 15., 16., 17.};
    const auto node_x0_host = Kokkos::View<double[2][7], Kokkos::HostSpace>::const_type(node_x0_host_data.data());
    const auto node_x0_mirror = Kokkos::create_mirror(node_x0);
    Kokkos::deep_copy(node_x0_mirror, node_x0_host);
    Kokkos::deep_copy(node_x0, node_x0_mirror);

    const auto node_u = Kokkos::View<double[2][7]>("node_u");
    constexpr auto node_u_host_data = std::array{0., 0., 0., 0., 0., 0., 0., 18., 19., 20., 21., 22., 23., 24.};
    const auto node_u_host = Kokkos::View<double[2][7], Kokkos::HostSpace>::const_type(node_u_host_data.data());
    const auto node_u_mirror = Kokkos::create_mirror(node_u);
    Kokkos::deep_copy(node_u_mirror, node_u_host);
    Kokkos::deep_copy(node_u, node_u_mirror);

    const auto node_udot = Kokkos::View<double[2][6]>("node_udot");
    constexpr auto node_udot_host_data = std::array{0., 0., 0., 0., 0., 0., 25., 26., 27., 28., 29., 30.};
    const auto node_udot_host = Kokkos::View<double[2][6], Kokkos::HostSpace>::const_type(node_udot_host_data.data());
    const auto node_udot_mirror = Kokkos::create_mirror(node_udot);
    Kokkos::deep_copy(node_udot_mirror, node_udot_host);
    Kokkos::deep_copy(node_udot, node_udot_mirror);

    const auto node_uddot = Kokkos::View<double[2][6]>("node_uddot");
    constexpr auto node_uddot_host_data = std::array{0., 0., 0., 0., 0., 0., 31., 32., 33., 34., 35., 36.};
    const auto node_uddot_host = Kokkos::View<double[2][6], Kokkos::HostSpace>::const_type(node_uddot_host_data.data());
    const auto node_uddot_mirror = Kokkos::create_mirror(node_uddot);
    Kokkos::deep_copy(node_uddot_mirror, node_uddot_host);
    Kokkos::deep_copy(node_uddot, node_uddot_mirror);

    const auto outputs = Kokkos::View<double[1][3]>("outputs");

    Kokkos::parallel_for("CalculateRevoluteJointOutput", 1, CalculateRevoluteJointOutput{target_node_index, axes, node_x0, node_u, node_udot, node_uddot, outputs});

    const auto outputs_mirror = Kokkos::create_mirror(outputs);
    Kokkos::deep_copy(outputs_mirror, outputs);

    constexpr auto outputs_exact_data = std::array{0., 358792., 433384.};
    const auto outputs_exact = Kokkos::View<double[1][3], Kokkos::HostSpace>::const_type(outputs_exact_data.data());

    for (auto i = 0U; i < 3U; ++i) {
        EXPECT_NEAR(outputs_mirror(0, i), outputs_exact(0, i), 1.e-15);
    }
}

}
