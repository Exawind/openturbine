#pragma once

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/math/vector_operations.hpp"

namespace openturbine {

/**
 * @brief Functor to calculate inertial forces in a beam/mass element
 *
 * This struct serves as a function object to compute inertial forces for beam and rigid body
 * elements, including both translational and rotational effects. The formulation follows the
 * "SO(3)-based GEBT Beam" chapter of the OpenTurbine theory documentation (Eq. 32):
 *
 * Inertial force vector = {
 *   {m * u_ddot + (omega_dot_tilde + omega_tilde * omega_tilde) * m * eta}
 *   {m * eta_tilde * u_ddot + rho * omega_dot + omega_tilde * rho * omega} }
 * }
 *
 * The forces are computed for each quadrature point (i_qp) of a given element (i_elem).
 */
struct CalculateInertialForces {
    using NoTranspose = KokkosBatched::Trans::NoTranspose;
    using GemmDefault = KokkosBatched::Algo::Gemm::Default;
    using GemvDefault = KokkosBlas::Algo::Gemv::Default;
    using Gemm = KokkosBatched::SerialGemm<NoTranspose, NoTranspose, GemmDefault>;
    using Gemv = KokkosBlas::SerialGemv<NoTranspose, GemvDefault>;
    size_t i_elem;                                         //< Element index
    Kokkos::View<double** [6][6]>::const_type qp_Muu_;     //< Mass matrix in inertial csys
    Kokkos::View<double** [3]>::const_type qp_u_ddot_;     //< Acceleration vector
    Kokkos::View<double** [3]>::const_type qp_omega_;      //< Angular velocity vector
    Kokkos::View<double** [3]>::const_type qp_omega_dot_;  //< Angular acceleration vector
    Kokkos::View<double** [3][3]>::const_type eta_tilde_;  //< Skew-symmetric matrix derived from eta
    Kokkos::View<double** [3][3]> omega_tilde_;      //< Skew-symmetric matrix from rot. velocity
    Kokkos::View<double** [3][3]> omega_dot_tilde_;  //< Skew-symmetric matrix from rot. acceleration
    Kokkos::View<double** [3][3]>::const_type rho_;  //< Rotational inertia part of mass matrix
    Kokkos::View<double** [3]>::const_type eta_;     //< Offset between mass center and elastic axis
    Kokkos::View<double** [6]> qp_FI_;               //< Inertial forces (computed in this functor)

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto Muu = Kokkos::subview(qp_Muu_, i_elem, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto u_ddot = Kokkos::subview(qp_u_ddot_, i_elem, i_qp, Kokkos::ALL);
        auto omega = Kokkos::subview(qp_omega_, i_elem, i_qp, Kokkos::ALL);
        auto omega_dot = Kokkos::subview(qp_omega_dot_, i_elem, i_qp, Kokkos::ALL);
        auto eta_tilde = Kokkos::subview(eta_tilde_, i_elem, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto omega_tilde = Kokkos::subview(omega_tilde_, i_elem, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto omega_dot_tilde =
            Kokkos::subview(omega_dot_tilde_, i_elem, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto rho = Kokkos::subview(rho_, i_elem, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto eta = Kokkos::subview(eta_, i_elem, i_qp, Kokkos::ALL);

        // Define temporary views for intermediate results
        auto v1 = Kokkos::Array<double, 3>{};
        auto V1 = View_3(v1.data());
        auto m1 = Kokkos::Array<double, 9>{};
        auto M1 = View_3x3(m1.data());
        auto FI = Kokkos::subview(qp_FI_, i_elem, i_qp, Kokkos::ALL);

        // Compute FI_1
        // FI_1 = m * u_ddot + (omega_dot_tilde + omega_tilde * omega_tilde) * m * eta
        VecTilde(omega, omega_tilde);
        VecTilde(omega_dot, omega_dot_tilde);
        auto FI_1 = Kokkos::subview(FI, Kokkos::make_pair(0, 3));
        auto m = Muu(0, 0);
        Gemm::invoke(m, omega_tilde, omega_tilde, 0., M1);
        KokkosBlas::serial_axpy(m, omega_dot_tilde, M1);
        Gemv::invoke(1., M1, eta, 0., FI_1);
        KokkosBlas::serial_axpy(m, u_ddot, FI_1);

        // Compute FI_2
        // FI_2 = m * eta_tilde * u_ddot + rho * omega_dot + omega_tilde * rho * omega
        auto FI_2 = Kokkos::subview(FI, Kokkos::make_pair(3, 6));
        KokkosBlas::serial_axpy(m, u_ddot, V1);
        Gemv::invoke(1., eta_tilde, V1, 0., FI_2);
        Gemv::invoke(1., rho, omega_dot, 1., FI_2);
        Gemm::invoke(1., omega_tilde, rho, 0., M1);
        Gemv::invoke(1., M1, omega, 1., FI_2);
    }
};

}  // namespace openturbine
