#pragma once

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace openturbine::tests {

inline auto create_shape_interp_OneNodeOneQP() {
    constexpr auto num_qp = 1;
    constexpr auto num_nodes = 1;
    constexpr auto num_entries = num_qp * num_nodes;
    auto shape_interp = Kokkos::View<double[1][num_nodes][num_qp]>("shape_interp");
    auto shape_interp_mirror = Kokkos::create_mirror(shape_interp);

    auto host_data = std::array<double, num_entries>{2.};
    auto shape_interp_host =
        Kokkos::View<double[1][num_nodes][num_qp], Kokkos::HostSpace>(host_data.data());
    Kokkos::deep_copy(shape_interp_mirror, shape_interp_host);
    Kokkos::deep_copy(shape_interp, shape_interp_mirror);
    return shape_interp;
}

inline auto create_shape_deriv_OneNodeOneQP() {
    constexpr auto num_qp = 1;
    constexpr auto num_nodes = 1;
    constexpr auto num_entries = num_qp * num_nodes;
    auto shape_deriv = Kokkos::View<double[1][num_nodes][num_qp]>("shape_deriv");
    auto shape_deriv_mirror = Kokkos::create_mirror(shape_deriv);

    auto host_data = std::array<double, num_entries>{4.};
    auto shape_deriv_host =
        Kokkos::View<double[1][num_nodes][num_qp], Kokkos::HostSpace>(host_data.data());
    Kokkos::deep_copy(shape_deriv_mirror, shape_deriv_host);
    Kokkos::deep_copy(shape_deriv, shape_deriv_mirror);
    return shape_deriv;
}

inline auto create_jacobian_OneQP() {
    constexpr auto num_qp = 1;
    constexpr auto num_entries = num_qp;
    auto jacobian = Kokkos::View<double[1][num_qp]>("jacobian");
    auto jacobian_mirror = Kokkos::create_mirror(jacobian);

    auto host_data = std::array<double, num_entries>{2.};
    auto jacobian_host = Kokkos::View<double[1][num_qp], Kokkos::HostSpace>(host_data.data());
    Kokkos::deep_copy(jacobian_mirror, jacobian_host);
    Kokkos::deep_copy(jacobian, jacobian_mirror);
    return jacobian;
}

inline auto create_shape_interp_OneNodeTwoQP() {
    constexpr auto num_qp = 2;
    constexpr auto num_nodes = 1;
    constexpr auto num_entries = num_qp * num_nodes;
    auto shape_interp = Kokkos::View<double[1][num_nodes][num_qp]>("shape_interp");
    auto shape_interp_mirror = Kokkos::create_mirror(shape_interp);

    auto host_data = std::array<double, num_entries>{2., 4.};
    auto shape_interp_host =
        Kokkos::View<double[1][num_nodes][num_qp], Kokkos::HostSpace>(host_data.data());
    Kokkos::deep_copy(shape_interp_mirror, shape_interp_host);
    Kokkos::deep_copy(shape_interp, shape_interp_mirror);
    return shape_interp;
}

inline auto create_shape_deriv_OneNodeTwoQP() {
    constexpr auto num_qp = 2;
    constexpr auto num_nodes = 1;
    constexpr auto num_entries = num_qp * num_nodes;
    auto shape_deriv = Kokkos::View<double[1][num_nodes][num_qp]>("shape_deriv");
    auto shape_deriv_mirror = Kokkos::create_mirror(shape_deriv);

    auto host_data = std::array<double, num_entries>{4., 9.};
    auto shape_deriv_host =
        Kokkos::View<double[1][num_nodes][num_qp], Kokkos::HostSpace>(host_data.data());
    Kokkos::deep_copy(shape_deriv_mirror, shape_deriv_host);
    Kokkos::deep_copy(shape_deriv, shape_deriv_mirror);
    return shape_deriv;
}

inline auto create_jacobian_TwoQP() {
    constexpr auto num_qp = 2;
    constexpr auto num_entries = num_qp;
    auto jacobian = Kokkos::View<double[1][num_qp]>("jacobian");
    auto jacobian_mirror = Kokkos::create_mirror(jacobian);

    auto host_data = std::array<double, num_entries>{2., 3.};
    auto jacobian_host = Kokkos::View<double[1][num_qp], Kokkos::HostSpace>(host_data.data());
    Kokkos::deep_copy(jacobian_mirror, jacobian_host);
    Kokkos::deep_copy(jacobian, jacobian_mirror);
    return jacobian;
}

inline auto create_shape_interp_TwoNodeTwoQP() {
    constexpr auto num_qp = 2;
    constexpr auto num_nodes = 2;
    constexpr auto num_entries = num_qp * num_nodes;
    auto shape_interp = Kokkos::View<double[1][num_nodes][num_qp]>("shape_interp");
    auto shape_interp_mirror = Kokkos::create_mirror(shape_interp);

    auto host_data = std::array<double, num_entries>{2., 3., 4., 5.};
    auto shape_interp_host =
        Kokkos::View<double[1][num_nodes][num_qp], Kokkos::HostSpace>(host_data.data());
    Kokkos::deep_copy(shape_interp_mirror, shape_interp_host);
    Kokkos::deep_copy(shape_interp, shape_interp_mirror);
    return shape_interp;
}

inline auto create_shape_deriv_TwoNodeTwoQP() {
    constexpr auto num_qp = 2;
    constexpr auto num_nodes = 2;
    constexpr auto num_entries = num_qp * num_nodes;
    auto shape_deriv = Kokkos::View<double[1][num_nodes][num_qp]>("shape_deriv");
    auto shape_deriv_mirror = Kokkos::create_mirror(shape_deriv);

    auto host_data = std::array<double, num_entries>{4., 9., 8., 15.};
    auto shape_deriv_host =
        Kokkos::View<double[1][num_nodes][num_qp], Kokkos::HostSpace>(host_data.data());
    Kokkos::deep_copy(shape_deriv_mirror, shape_deriv_host);
    Kokkos::deep_copy(shape_deriv, shape_deriv_mirror);
    return shape_deriv;
}

}  // namespace openturbine::tests