#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "create_view.hpp"
#include "state/update_algorithmic_acceleration.hpp"

namespace openturbine::tests {
TEST(UpdateAlgorithmicAcceleration, OneNode) {
    constexpr auto alpha_f = 0.5;
    constexpr auto alpha_m = 0.75;

    const auto vd = CreateView<double[1][6]>("vd", std::array{7., 8., 9., 10., 11., 12.});

    const auto acceleration = Kokkos::View<double[1][6]>("acceleration");

    Kokkos::parallel_for(
        "UpdateAlgorithmicAcceleration", 1,
        UpdateAlgorithmicAcceleration{acceleration, vd, alpha_f, alpha_m}
    );

    constexpr auto acceleration_exact_data = std::array{14., 16., 18., 20., 22., 24.};
    const auto acceleration_exact =
        Kokkos::View<double[1][6], Kokkos::HostSpace>::const_type(acceleration_exact_data.data());
    const auto acceleration_mirror =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), acceleration);
    for (auto j = 0U; j < 6U; ++j) {
        EXPECT_EQ(acceleration_mirror(0, j), acceleration_exact(0, j));
    }
}
}  // namespace openturbine::tests
