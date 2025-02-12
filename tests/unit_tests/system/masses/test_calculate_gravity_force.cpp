#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "system/masses/calculate_gravity_force.hpp"
#include "test_calculate.hpp"

namespace openturbine::tests {

struct ExecuteCalculateGravityForce {
    double mass;
    Kokkos::View<double[3]>::const_type gravity;
    Kokkos::View<double[3][3]>::const_type eta_tilde;
    Kokkos::View<double[6]> FG;

    KOKKOS_FUNCTION
    void operator()(size_t) const { masses::CalculateGravityForce(mass, gravity, eta_tilde, FG); }
};

TEST(CalculateGravityForceTestsMasses, OneNode) {
    constexpr auto mass = 1.;

    const auto eta_tilde = Kokkos::View<double[3][3]>("eta_tilde");
    constexpr auto eta_tilde_data = std::array{37., 38., 39., 40., 41., 42., 43., 44., 45.};
    const auto eta_tilde_host =
        Kokkos::View<const double[3][3], Kokkos::HostSpace>(eta_tilde_data.data());
    const auto eta_tilde_mirror = Kokkos::create_mirror(eta_tilde);
    Kokkos::deep_copy(eta_tilde_mirror, eta_tilde_host);
    Kokkos::deep_copy(eta_tilde, eta_tilde_mirror);

    const auto gravity = Kokkos::View<double[3]>("gravity");
    constexpr auto gravity_data = std::array{46., 47., 48.};
    const auto gravity_host = Kokkos::View<const double[3], Kokkos::HostSpace>(gravity_data.data());
    const auto gravity_mirror = Kokkos::create_mirror(gravity);
    Kokkos::deep_copy(gravity_mirror, gravity_host);
    Kokkos::deep_copy(gravity, gravity_mirror);

    const auto FG = Kokkos::View<double[6]>("FG");

    Kokkos::parallel_for(
        "CalculateGravityForce", 1, ExecuteCalculateGravityForce{mass, gravity, eta_tilde, FG}
    );

    constexpr auto FG_exact_data = std::array{46., 47., 48., 5360., 5783., 6206.};
    const auto FG_exact = Kokkos::View<const double[6], Kokkos::HostSpace>(FG_exact_data.data());

    const auto FG_mirror = Kokkos::create_mirror(FG);
    Kokkos::deep_copy(FG_mirror, FG);
    CompareWithExpected(FG_mirror, FG_exact);
}

}  // namespace openturbine::tests
