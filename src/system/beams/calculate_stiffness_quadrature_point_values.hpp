#pragma once

#include <KokkosBatched_Copy_Decl.hpp>
#include <Kokkos_Core.hpp>

#include "calculate_Ouu.hpp"
#include "calculate_Puu.hpp"
#include "calculate_Quu.hpp"
#include "calculate_force_FC.hpp"
#include "calculate_force_FD.hpp"
#include "calculate_strain.hpp"
#include "calculate_temporary_variables.hpp"
#include "interpolate_to_quadrature_point_for_stiffness.hpp"
#include "system/masses/rotate_section_matrix.hpp"

namespace openturbine::beams {

template <typename DeviceType>
struct CalculateStiffnessQuadraturePointValues {
    size_t element;

    typename Kokkos::View<double*, DeviceType>::const_type qp_jacobian;
    typename Kokkos::View<double**, Kokkos::LayoutLeft, DeviceType>::const_type shape_interp;
    typename Kokkos::View<double**, Kokkos::LayoutLeft, DeviceType>::const_type shape_deriv;
    typename Kokkos::View<double** [4], DeviceType>::const_type qp_r0;
    typename Kokkos::View<double** [3], DeviceType>::const_type qp_x0_prime;
    typename Kokkos::View<double** [6][6], DeviceType>::const_type qp_Cstar;
    typename Kokkos::View<double* [7], DeviceType>::const_type node_u;

    Kokkos::View<double* [6], DeviceType> qp_Fc;
    Kokkos::View<double* [6], DeviceType> qp_Fd;
    Kokkos::View<double* [6][6], DeviceType> qp_Cuu;
    Kokkos::View<double* [6][6], DeviceType> qp_Ouu;
    Kokkos::View<double* [6][6], DeviceType> qp_Puu;
    Kokkos::View<double* [6][6], DeviceType> qp_Quu;

    KOKKOS_FUNCTION
    void operator()(size_t qp) const {
        const auto r0_data = Kokkos::Array<double, 4>{
            qp_r0(element, qp, 0), qp_r0(element, qp, 1), qp_r0(element, qp, 2),
            qp_r0(element, qp, 3)
        };
        const auto x0_prime_data = Kokkos::Array<double, 3>{
            qp_x0_prime(element, qp, 0), qp_x0_prime(element, qp, 1), qp_x0_prime(element, qp, 2)
        };
        auto xr_data = Kokkos::Array<double, 4>{};
        auto u_data = Kokkos::Array<double, 3>{};
        auto u_prime_data = Kokkos::Array<double, 3>{};
        auto r_data = Kokkos::Array<double, 4>{};
        auto r_prime_data = Kokkos::Array<double, 4>{};
        auto Cstar_data = Kokkos::Array<double, 36>{};

        auto strain_data = Kokkos::Array<double, 6>{};
        auto x0pupSS_data = Kokkos::Array<double, 9>{};
        auto M_tilde_data = Kokkos::Array<double, 9>{};
        auto N_tilde_data = Kokkos::Array<double, 9>{};
        auto FC_data = Kokkos::Array<double, 6>{};
        auto FD_data = Kokkos::Array<double, 6>{};
        auto Cuu_data = Kokkos::Array<double, 36>{};
        auto Ouu_data = Kokkos::Array<double, 36>{};
        auto Puu_data = Kokkos::Array<double, 36>{};
        auto Quu_data = Kokkos::Array<double, 36>{};

        const auto r0 = typename Kokkos::View<double[4], DeviceType>::const_type(r0_data.data());
        const auto x0_prime =
            typename Kokkos::View<double[3], DeviceType>::const_type(x0_prime_data.data());
        const auto xr = Kokkos::View<double[4], DeviceType>(xr_data.data());
        const auto u = Kokkos::View<double[3], DeviceType>(u_data.data());
        const auto u_prime = Kokkos::View<double[3], DeviceType>(u_prime_data.data());
        const auto r = Kokkos::View<double[4], DeviceType>(r_data.data());
        const auto r_prime = Kokkos::View<double[4], DeviceType>(r_prime_data.data());

        const auto strain = Kokkos::View<double[6], DeviceType>(strain_data.data());
        const auto x0pupSS = Kokkos::View<double[3][3], DeviceType>(x0pupSS_data.data());
        const auto M_tilde = Kokkos::View<double[3][3], DeviceType>(M_tilde_data.data());
        const auto N_tilde = Kokkos::View<double[3][3], DeviceType>(N_tilde_data.data());
        const auto FC = Kokkos::View<double[6], DeviceType>(FC_data.data());
        const auto FD = Kokkos::View<double[6], DeviceType>(FD_data.data());
        const auto Cstar = Kokkos::View<double[6][6], DeviceType>(Cstar_data.data());
        const auto Cuu = Kokkos::View<double[6][6], DeviceType>(Cuu_data.data());
        const auto Ouu = Kokkos::View<double[6][6], DeviceType>(Ouu_data.data());
        const auto Puu = Kokkos::View<double[6][6], DeviceType>(Puu_data.data());
        const auto Quu = Kokkos::View<double[6][6], DeviceType>(Quu_data.data());

        KokkosBatched::SerialCopy<>::invoke(
            Kokkos::subview(qp_Cstar, element, qp, Kokkos::ALL, Kokkos::ALL), Cstar
        );

        beams::InterpolateToQuadraturePointForStiffness<DeviceType>(
            qp_jacobian(qp), Kokkos::subview(shape_interp, Kokkos::ALL, qp),
            Kokkos::subview(shape_deriv, Kokkos::ALL, qp), node_u, u, r, u_prime, r_prime
        );
        QuaternionCompose(r, r0, xr);

        masses::RotateSectionMatrix<DeviceType>(xr, Cstar, Cuu);

        beams::CalculateStrain<DeviceType>(x0_prime, u_prime, r, r_prime, strain);
        beams::CalculateTemporaryVariables<DeviceType>(x0_prime, u_prime, x0pupSS);
        beams::CalculateForceFC<DeviceType>(Cuu, strain, FC);
        beams::CalculateForceFD<DeviceType>(x0pupSS, FC, FD);

        VecTilde(Kokkos::subview(FC, Kokkos::make_pair(0, 3)), N_tilde);
        VecTilde(Kokkos::subview(FC, Kokkos::make_pair(3, 6)), M_tilde);

        beams::CalculateOuu<DeviceType>(Cuu, x0pupSS, M_tilde, N_tilde, Ouu);
        beams::CalculatePuu<DeviceType>(Cuu, x0pupSS, N_tilde, Puu);
        beams::CalculateQuu<DeviceType>(Cuu, x0pupSS, N_tilde, Quu);

        KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose, 1>::invoke(
            FC, Kokkos::subview(qp_Fc, qp, Kokkos::ALL)
        );
        KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose, 1>::invoke(
            FD, Kokkos::subview(qp_Fd, qp, Kokkos::ALL)
        );

        KokkosBatched::SerialCopy<>::invoke(
            Cuu, Kokkos::subview(qp_Cuu, qp, Kokkos::ALL, Kokkos::ALL)
        );
        KokkosBatched::SerialCopy<>::invoke(
            Ouu, Kokkos::subview(qp_Ouu, qp, Kokkos::ALL, Kokkos::ALL)
        );
        KokkosBatched::SerialCopy<>::invoke(
            Puu, Kokkos::subview(qp_Puu, qp, Kokkos::ALL, Kokkos::ALL)
        );
        KokkosBatched::SerialCopy<>::invoke(
            Quu, Kokkos::subview(qp_Quu, qp, Kokkos::ALL, Kokkos::ALL)
        );
    }
};

}  // namespace openturbine::beams
