#pragma once

#include <KokkosBatched_Copy_Decl.hpp>
#include <Kokkos_Core.hpp>

#include "interpolate_to_quadrature_point_for_inertia.hpp"
#include "system/beams/integrate_inertia_matrix.hpp"
#include "system/beams/integrate_residual_vector.hpp"
#include "system/beams/integrate_stiffness_matrix.hpp"
#include "system/masses/calculate_gravity_force.hpp"
#include "system/masses/calculate_gyroscopic_matrix.hpp"
#include "system/masses/calculate_inertia_stiffness_matrix.hpp"
#include "system/masses/calculate_inertial_force.hpp"
#include "system/masses/calculate_mass_matrix_components.hpp"
#include "system/masses/rotate_section_matrix.hpp"

namespace openturbine::beams {

template <typename DeviceType>
struct CalculateInertialQuadraturePointValues {
    size_t i_elem;

    typename Kokkos::View<double**, Kokkos::LayoutLeft, DeviceType>::const_type shape_interp;
    typename Kokkos::View<double[3], DeviceType>::const_type gravity;
    typename Kokkos::View<double** [4], DeviceType>::const_type qp_r0;
    typename Kokkos::View<double** [6][6], DeviceType>::const_type qp_Mstar;
    typename Kokkos::View<double* [7], DeviceType>::const_type node_u;
    typename Kokkos::View<double* [6], DeviceType>::const_type node_u_dot;
    typename Kokkos::View<double* [6], DeviceType>::const_type node_u_ddot;

    Kokkos::View<double* [6], DeviceType> qp_Fi;
    Kokkos::View<double* [6], DeviceType> qp_Fg;
    Kokkos::View<double* [6][6], DeviceType> qp_Muu;
    Kokkos::View<double* [6][6], DeviceType> qp_Guu;
    Kokkos::View<double* [6][6], DeviceType> qp_Kuu;

    KOKKOS_FUNCTION
    void operator()(size_t i_qp) const {
        const auto r0_data = Kokkos::Array<double, 4>{
            qp_r0(i_elem, i_qp, 0), qp_r0(i_elem, i_qp, 1), qp_r0(i_elem, i_qp, 2),
            qp_r0(i_elem, i_qp, 3)
        };
        auto r_data = Kokkos::Array<double, 4>{};
        auto xr_data = Kokkos::Array<double, 4>{};
        auto u_ddot_data = Kokkos::Array<double, 3>{};
        auto omega_data = Kokkos::Array<double, 3>{};
        auto omega_dot_data = Kokkos::Array<double, 3>{};
        auto Mstar_data = Kokkos::Array<double, 36>{};

        auto eta_data = Kokkos::Array<double, 3>{};
        auto eta_tilde_data = Kokkos::Array<double, 9>{};
        auto rho_data = Kokkos::Array<double, 9>{};
        auto omega_tilde_data = Kokkos::Array<double, 9>{};
        auto omega_dot_tilde_data = Kokkos::Array<double, 9>{};
        auto FI_data = Kokkos::Array<double, 6>{};
        auto FG_data = Kokkos::Array<double, 6>{};
        auto Muu_data = Kokkos::Array<double, 36>{};
        auto Guu_data = Kokkos::Array<double, 36>{};
        auto Kuu_data = Kokkos::Array<double, 36>{};

        const auto r0 = typename Kokkos::View<double[4], DeviceType>::const_type(r0_data.data());
        const auto r = Kokkos::View<double[4], DeviceType>(r_data.data());
        const auto xr = Kokkos::View<double[4], DeviceType>(xr_data.data());
        const auto u_ddot = Kokkos::View<double[3], DeviceType>(u_ddot_data.data());
        const auto omega = Kokkos::View<double[3], DeviceType>(omega_data.data());
        const auto omega_dot = Kokkos::View<double[3], DeviceType>(omega_dot_data.data());

        const auto eta = Kokkos::View<double[3], DeviceType>(eta_data.data());
        const auto eta_tilde = Kokkos::View<double[3][3], DeviceType>(eta_tilde_data.data());
        const auto rho = Kokkos::View<double[3][3], DeviceType>(rho_data.data());
        const auto omega_tilde = Kokkos::View<double[3][3], DeviceType>(omega_tilde_data.data());
        const auto omega_dot_tilde = Kokkos::View<double[3][3], DeviceType>(omega_dot_tilde_data.data());
        const auto FI = Kokkos::View<double[6], DeviceType>(FI_data.data());
        const auto FG = Kokkos::View<double[6], DeviceType>(FG_data.data());
        const auto Mstar = Kokkos::View<double[6][6], DeviceType>(Mstar_data.data());
        const auto Muu = Kokkos::View<double[6][6], DeviceType>(Muu_data.data());
        const auto Guu = Kokkos::View<double[6][6], DeviceType>(Guu_data.data());
        const auto Kuu = Kokkos::View<double[6][6], DeviceType>(Kuu_data.data());

        KokkosBatched::SerialCopy<>::invoke(
            Kokkos::subview(qp_Mstar, i_elem, i_qp, Kokkos::ALL, Kokkos::ALL), Mstar
        );
        beams::InterpolateToQuadraturePointForInertia<DeviceType>(
            Kokkos::subview(shape_interp, Kokkos::ALL, i_qp), node_u, node_u_dot, node_u_ddot, r,
            u_ddot, omega, omega_dot
        );

        QuaternionCompose(r, r0, xr);
        masses::RotateSectionMatrix<DeviceType>(xr, Mstar, Muu);

        const auto mass = Muu(0, 0);
        masses::CalculateEta<DeviceType>(Muu, eta);
        VecTilde(eta, eta_tilde);
        masses::CalculateRho<DeviceType>(Muu, rho);

        VecTilde(omega, omega_tilde);
        VecTilde(omega_dot, omega_dot_tilde);

        masses::CalculateInertialForce<DeviceType>(
            mass, u_ddot, omega, omega_dot, eta, eta_tilde, rho, omega_tilde, omega_dot_tilde, FI
        );
        masses::CalculateGravityForce<DeviceType>(mass, gravity, eta_tilde, FG);

        masses::CalculateGyroscopicMatrix<DeviceType>(mass, omega, eta, rho, omega_tilde, Guu);
        masses::CalculateInertiaStiffnessMatrix<DeviceType>(
            mass, u_ddot, omega, omega_dot, eta, rho, omega_tilde, omega_dot_tilde, Kuu
        );

        KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose, 1>::invoke(
            FI, Kokkos::subview(qp_Fi, i_qp, Kokkos::ALL)
        );
        KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose, 1>::invoke(
            FG, Kokkos::subview(qp_Fg, i_qp, Kokkos::ALL)
        );
        KokkosBatched::SerialCopy<>::invoke(
            Muu, Kokkos::subview(qp_Muu, i_qp, Kokkos::ALL, Kokkos::ALL)
        );
        KokkosBatched::SerialCopy<>::invoke(
            Guu, Kokkos::subview(qp_Guu, i_qp, Kokkos::ALL, Kokkos::ALL)
        );
        KokkosBatched::SerialCopy<>::invoke(
            Kuu, Kokkos::subview(qp_Kuu, i_qp, Kokkos::ALL, Kokkos::ALL)
        );
    }
};

}  // namespace openturbine::beams
