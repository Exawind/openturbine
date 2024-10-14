
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "src/state/update_static_prediction.hpp"

namespace openturbine::tests {
TEST(UpdateStaticPrediction, TwoNodes) {
    constexpr auto h = 2.;
    constexpr auto beta_prime = 20.;
    constexpr auto gamma_prime = 50.;

    const auto x_delta = Kokkos::View<double[12]>("x_delta");
    constexpr auto x_delta_host_data =
        std::array{2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 22., 24.};
    const auto x_delta_host =
        Kokkos::View<double[12], Kokkos::HostSpace>::const_type(x_delta_host_data.data());
    const auto x_delta_mirror = Kokkos::create_mirror(x_delta);
    Kokkos::deep_copy(x_delta_mirror, x_delta_host);
    Kokkos::deep_copy(x_delta, x_delta_mirror);

    const auto q_delta = Kokkos::View<double[2][6]>("q_delta");

    Kokkos::parallel_for(
        "UpdateStaticPrediction", 2,
        UpdateStaticPrediction{h, beta_prime, gamma_prime, x_delta, q_delta}
    );

    constexpr auto q_delta_exact_data =
        std::array{1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
    const auto q_delta_exact =
        Kokkos::View<double[2][6], Kokkos::HostSpace>::const_type(q_delta_exact_data.data());
    const auto q_delta_mirror = Kokkos::create_mirror(q_delta);
    Kokkos::deep_copy(q_delta_mirror, q_delta);
    for (auto i = 0U; i < 6U; ++i) {
        EXPECT_NEAR(q_delta_mirror(0, i), q_delta_exact(0, i), 1.e-14);
    }
    for (auto i = 0U; i < 6U; ++i) {
        EXPECT_NEAR(q_delta_mirror(1, i), q_delta_exact(1, i), 1.e-14);
    }
}

}  // namespace openturbine::tests
