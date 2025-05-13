#pragma once

#include <Kokkos_Core.hpp>

#include "system/masses/calculate_gravity_force.hpp"
#include "system/masses/calculate_gyroscopic_matrix.hpp"
#include "system/masses/calculate_inertia_stiffness_matrix.hpp"
#include "system/masses/calculate_inertial_force.hpp"
#include "system/masses/calculate_mass_matrix_components.hpp"
#include "system/masses/rotate_section_matrix.hpp"

namespace openturbine::masses {

template <typename DeviceType>
struct CalculateQuadraturePointValues {
    double beta_prime;
    double gamma_prime;

    typename Kokkos::View<double* [7], DeviceType>::const_type Q;
    typename Kokkos::View<double* [6], DeviceType>::const_type V;
    typename Kokkos::View<double* [6], DeviceType>::const_type A;

    typename Kokkos::View<double* [6][6], DeviceType>::const_type tangent;
    typename Kokkos::View<size_t*, DeviceType>::const_type node_state_indices;
    typename Kokkos::View<double[3], DeviceType>::const_type gravity;
    typename Kokkos::View<double* [6][6], DeviceType>::const_type qp_Mstar;
    typename Kokkos::View<double* [7], DeviceType>::const_type node_x0;

    Kokkos::View<double* [6], DeviceType> residual_vector_terms;
    Kokkos::View<double* [6][6], DeviceType> system_matrix_terms;

    KOKKOS_FUNCTION
    void operator()(size_t i_elem) const {
        const auto index = node_state_indices(i_elem);

        // Allocate scratch views
        auto Mstar_data = Kokkos::Array<double, 36>{};
        const auto x0_data =
            Kokkos::Array<double, 3>{node_x0(i_elem, 0), node_x0(i_elem, 1), node_x0(i_elem, 2)};
        const auto r0_data = Kokkos::Array<double, 4>{
            node_x0(i_elem, 3), node_x0(i_elem, 4), node_x0(i_elem, 5), node_x0(i_elem, 6)
        };
        const auto u_data = Kokkos::Array<double, 3>{Q(index, 0), Q(index, 1), Q(index, 2)};
        const auto r_data =
            Kokkos::Array<double, 4>{Q(index, 3), Q(index, 4), Q(index, 5), Q(index, 6)};
        auto xr_data = Kokkos::Array<double, 4>{};
        const auto u_ddot_data = Kokkos::Array<double, 3>{A(index, 0), A(index, 1), A(index, 2)};
        const auto omega_data = Kokkos::Array<double, 3>{V(index, 3), V(index, 4), V(index, 5)};
        const auto omega_dot_data = Kokkos::Array<double, 3>{A(index, 3), A(index, 4), A(index, 5)};
        auto Muu_data = Kokkos::Array<double, 36>{};
        auto Fg_data = Kokkos::Array<double, 6>{};
        auto eta_data = Kokkos::Array<double, 3>{};
        auto eta_tilde_data = Kokkos::Array<double, 9>{};
        auto rho_data = Kokkos::Array<double, 9>{};
        auto omega_tilde_data = Kokkos::Array<double, 9>{};
        auto omega_dot_tilde_data = Kokkos::Array<double, 9>{};
        auto Fi_data = Kokkos::Array<double, 6>{};
        auto Guu_data = Kokkos::Array<double, 36>{};
        auto Kuu_data = Kokkos::Array<double, 36>{};
        auto T_data = Kokkos::Array<double, 36>{};
        auto STpI_data = Kokkos::Array<double, 36>{};

        // Set up Views
        const auto Mstar = Kokkos::View<double[6][6], DeviceType>(Mstar_data.data());
        const auto x0 = typename Kokkos::View<double[3], DeviceType>::const_type(x0_data.data());
        const auto r0 = typename Kokkos::View<double[4], DeviceType>::const_type(r0_data.data());
        const auto u = typename Kokkos::View<double[3], DeviceType>::const_type(u_data.data());
        const auto r = typename Kokkos::View<double[4], DeviceType>::const_type(r_data.data());
        const auto xr = Kokkos::View<double[4], DeviceType>(xr_data.data());
        const auto u_ddot = typename Kokkos::View<double[3], DeviceType>::const_type(u_ddot_data.data());
        const auto omega = typename Kokkos::View<double[3], DeviceType>::const_type(omega_data.data());
        const auto omega_dot = typename Kokkos::View<double[3], DeviceType>::const_type(omega_dot_data.data());
        auto Muu = Kokkos::View<double[6][6], DeviceType>(Muu_data.data());
        auto Fg = Kokkos::View<double[6], DeviceType>(Fg_data.data());
        auto eta = Kokkos::View<double[3], DeviceType>(eta_data.data());
        auto eta_tilde = Kokkos::View<double[3][3], DeviceType>(eta_tilde_data.data());
        auto rho = Kokkos::View<double[3][3], DeviceType>(rho_data.data());
        auto omega_tilde = Kokkos::View<double[3][3], DeviceType>(omega_tilde_data.data());
        auto omega_dot_tilde = Kokkos::View<double[3][3], DeviceType>(omega_dot_tilde_data.data());
        auto Fi = Kokkos::View<double[6], DeviceType>(Fi_data.data());
        auto Guu = Kokkos::View<double[6][6], DeviceType>(Guu_data.data());
        auto Kuu = Kokkos::View<double[6][6], DeviceType>(Kuu_data.data());
        auto T = Kokkos::View<double[6][6], DeviceType>(T_data.data());
        auto STpI = Kokkos::View<double[6][6], DeviceType>(STpI_data.data());

        // Do the math
        KokkosBatched::SerialCopy<>::invoke(
            Kokkos::subview(qp_Mstar, i_elem, Kokkos::ALL, Kokkos::ALL), Mstar
        );
        QuaternionCompose(r, r0, xr);
        VecTilde(omega, omega_tilde);
        VecTilde(omega_dot, omega_dot_tilde);

        RotateSectionMatrix<DeviceType>(xr, Mstar, Muu);

        const auto mass = Muu(0, 0);
        CalculateEta<DeviceType>(Muu, eta);
        VecTilde(eta, eta_tilde);
        CalculateRho<DeviceType>(Muu, rho);

        CalculateGravityForce<DeviceType>(mass, gravity, eta_tilde, Fg);
        CalculateInertialForce<DeviceType>(
            mass, u_ddot, omega, omega_dot, eta, eta_tilde, rho, omega_tilde, omega_dot_tilde, Fi
        );

        CalculateGyroscopicMatrix<DeviceType>(mass, omega, eta, rho, omega_tilde, Guu);
        CalculateInertiaStiffnessMatrix<DeviceType>(
            mass, u_ddot, omega, omega_dot, eta, rho, omega_tilde, omega_dot_tilde, Kuu
        );

        // Contribute terms to main matrices
        for (auto i = 0U; i < 6U; ++i) {
            residual_vector_terms(i_elem, i) = Fi(i) - Fg(i);
        }
        KokkosBatched::SerialCopy<>::invoke(
            Kokkos::subview(tangent, index, Kokkos::ALL, Kokkos::ALL), T
        );
        for (auto i = 0U; i < 6U; ++i) {
            for (auto j = 0U; j < 6U; ++j) {
                STpI(i, j) = beta_prime * Muu(i, j) + gamma_prime * Guu(i, j);
            }
        }
        KokkosBatched::SerialGemm<
            KokkosBatched::Trans::NoTranspose, KokkosBatched::Trans::NoTranspose,
            KokkosBatched::Algo::Gemm::Default>::invoke(1., Kuu, T, 1., STpI);

        KokkosBatched::SerialCopy<>::invoke(
            STpI, Kokkos::subview(system_matrix_terms, i_elem, Kokkos::ALL, Kokkos::ALL)
        );
    }
};

}  // namespace openturbine::masses
