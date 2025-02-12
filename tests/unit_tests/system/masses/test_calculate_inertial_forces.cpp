#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "system/masses/calculate_inertial_force.hpp"
#include "test_calculate.hpp"

namespace openturbine::tests {

struct ExecuteCalculateInertialForces {
    double mass;
    Kokkos::View<double[3]>::const_type u_ddot;
    Kokkos::View<double[3]>::const_type omega;
    Kokkos::View<double[3]>::const_type omega_dot;
    Kokkos::View<double[3][3]>::const_type eta_tilde;
    Kokkos::View<double[3][3]> omega_tilde;
    Kokkos::View<double[3][3]> omega_dot_tilde;
    Kokkos::View<double[3][3]>::const_type rho;
    Kokkos::View<double[3]>::const_type eta;
    Kokkos::View<double[6]> FI;

    KOKKOS_FUNCTION
    void operator()(size_t) const {
        masses::CalculateInertialForce(
            mass, u_ddot, omega, omega_dot, eta, eta_tilde, rho, omega_tilde, omega_dot_tilde, FI
        );
    }
};

TEST(CalculateInertialForcesTestsMasses, OneNode) {
    double mass = 1.;

    const auto u_ddot = Kokkos::View<double[3]>("u_ddot");
    constexpr auto u_ddot_data = std::array{37., 38., 39.};
    const auto u_ddot_host = Kokkos::View<const double[3], Kokkos::HostSpace>(u_ddot_data.data());
    const auto u_ddot_mirror = Kokkos::create_mirror(u_ddot);
    Kokkos::deep_copy(u_ddot_mirror, u_ddot_host);
    Kokkos::deep_copy(u_ddot, u_ddot_mirror);

    const auto omega = Kokkos::View<double[3]>("omega");
    constexpr auto omega_data = std::array{40., 41., 42.};
    const auto omega_host = Kokkos::View<const double[3], Kokkos::HostSpace>(omega_data.data());
    const auto omega_mirror = Kokkos::create_mirror(omega);
    Kokkos::deep_copy(omega_mirror, omega_host);
    Kokkos::deep_copy(omega, omega_mirror);

    const auto omega_dot = Kokkos::View<double[3]>("omega_dot");
    constexpr auto omega_dot_data = std::array{43., 44., 45.};
    const auto omega_dot_host =
        Kokkos::View<const double[3], Kokkos::HostSpace>(omega_dot_data.data());
    const auto omega_dot_mirror = Kokkos::create_mirror(omega_dot);
    Kokkos::deep_copy(omega_dot_mirror, omega_dot_host);
    Kokkos::deep_copy(omega_dot, omega_dot_mirror);

    const auto eta_tilde = Kokkos::View<double[3][3]>("eta_tilde");
    constexpr auto eta_tilde_data = std::array{46., 47., 48., 49., 50., 51., 52., 53., 54.};
    const auto eta_tilde_host =
        Kokkos::View<const double[3][3], Kokkos::HostSpace>(eta_tilde_data.data());
    const auto eta_tilde_mirror = Kokkos::create_mirror(eta_tilde);
    Kokkos::deep_copy(eta_tilde_mirror, eta_tilde_host);
    Kokkos::deep_copy(eta_tilde, eta_tilde_mirror);

    const auto rho = Kokkos::View<double[3][3]>("rho");
    const auto rho_data = std::array{55., 56., 57., 58., 59., 60., 61., 62., 63.};
    const auto rho_host = Kokkos::View<const double[3][3], Kokkos::HostSpace>(rho_data.data());
    const auto rho_mirror = Kokkos::create_mirror(rho);
    Kokkos::deep_copy(rho_mirror, rho_host);
    Kokkos::deep_copy(rho, rho_mirror);

    const auto eta = Kokkos::View<double[3]>("eta");
    const auto eta_data = std::array{64., 65., 66.};
    const auto eta_host = Kokkos::View<const double[3], Kokkos::HostSpace>(eta_data.data());
    const auto eta_mirror = Kokkos::create_mirror(eta);
    Kokkos::deep_copy(eta_mirror, eta_host);
    Kokkos::deep_copy(eta, eta_mirror);

    const auto omega_tilde = Kokkos::View<double[3][3]>("omega_tilde");
    constexpr auto omega_tilde_data = std::array{0., -42., 41., 42., 0., -40., -41., 40., 0.};
    const auto omega_tilde_host =
        Kokkos::View<const double[3][3], Kokkos::HostSpace>(omega_tilde_data.data());
    const auto omega_tilde_mirror = Kokkos::create_mirror(omega_tilde);
    Kokkos::deep_copy(omega_tilde_mirror, omega_tilde_host);
    Kokkos::deep_copy(omega_tilde, omega_tilde_mirror);

    const auto omega_dot_tilde = Kokkos::View<double[3][3]>("omega_dot_tilde");
    constexpr auto omega_dot_tilde_data = std::array{0., -45., 44., 45., 0., -43., -44., 43., 0.};
    const auto omega_dot_tilde_host =
        Kokkos::View<const double[3][3], Kokkos::HostSpace>(omega_dot_tilde_data.data());
    const auto omega_dot_tilde_mirror = Kokkos::create_mirror(omega_dot_tilde);
    Kokkos::deep_copy(omega_dot_tilde_mirror, omega_dot_tilde_host);
    Kokkos::deep_copy(omega_dot_tilde, omega_dot_tilde_mirror);

    const auto FI = Kokkos::View<double[6]>("FI");

    Kokkos::parallel_for(
        "CalculateInertialForces", 1,
        ExecuteCalculateInertialForces{
            mass, u_ddot, omega, omega_dot, eta_tilde, omega_tilde, omega_dot_tilde, rho, eta, FI
        }
    );

    constexpr auto FI_exact_data = std::array{-2984., 32., 2922., 20624., -2248., 22100.};
    const auto FI_exact = Kokkos::View<const double[6], Kokkos::HostSpace>(FI_exact_data.data());

    const auto FI_mirror = Kokkos::create_mirror(FI);
    Kokkos::deep_copy(FI_mirror, FI);
    CompareWithExpected(FI_mirror, FI_exact);
}

}  // namespace openturbine::tests
