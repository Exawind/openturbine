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

namespace openturbine::beams {

struct CalculateStiffnessQuadraturePointValues {
    size_t i_elem;

    Kokkos::View<double*>::const_type qp_jacobian;
    Kokkos::View<double**, Kokkos::LayoutLeft>::const_type shape_interp;
    Kokkos::View<double**, Kokkos::LayoutLeft>::const_type shape_deriv;
    Kokkos::View<double** [4]>::const_type qp_r0;
    Kokkos::View<double** [3]>::const_type qp_x0_prime;
    Kokkos::View<double** [6][6]>::const_type qp_Cstar;
    Kokkos::View<double* [7]>::const_type node_u;

    Kokkos::View<double* [6]> qp_Fc;
    Kokkos::View<double* [6]> qp_Fd;
    Kokkos::View<double* [6][6]> qp_Cuu;
    Kokkos::View<double* [6][6]> qp_Ouu;
    Kokkos::View<double* [6][6]> qp_Puu;
    Kokkos::View<double* [6][6]> qp_Quu;

    KOKKOS_FUNCTION
    void operator()(size_t i_qp) const {
        const auto r0_data = Kokkos::Array<double, 4>{
            qp_r0(i_elem, i_qp, 0), qp_r0(i_elem, i_qp, 1), qp_r0(i_elem, i_qp, 2),
            qp_r0(i_elem, i_qp, 3)
        };
        const auto x0_prime_data = Kokkos::Array<double, 3>{
            qp_x0_prime(i_elem, i_qp, 0), qp_x0_prime(i_elem, i_qp, 1), qp_x0_prime(i_elem, i_qp, 2)
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

        const auto r0 = Kokkos::View<double[4]>::const_type(r0_data.data());
        const auto x0_prime = Kokkos::View<double[3]>::const_type(x0_prime_data.data());
        const auto xr = Kokkos::View<double[4]>(xr_data.data());
        const auto u = Kokkos::View<double[3]>(u_data.data());
        const auto u_prime = Kokkos::View<double[3]>(u_prime_data.data());
        const auto r = Kokkos::View<double[4]>(r_data.data());
        const auto r_prime = Kokkos::View<double[4]>(r_prime_data.data());

        const auto strain = Kokkos::View<double[6]>(strain_data.data());
        const auto x0pupSS = Kokkos::View<double[3][3]>(x0pupSS_data.data());
        const auto M_tilde = Kokkos::View<double[3][3]>(M_tilde_data.data());
        const auto N_tilde = Kokkos::View<double[3][3]>(N_tilde_data.data());
        const auto FC = Kokkos::View<double[6]>(FC_data.data());
        const auto FD = Kokkos::View<double[6]>(FD_data.data());
        const auto Cstar = Kokkos::View<double[6][6]>(Cstar_data.data());
        const auto Cuu = Kokkos::View<double[6][6]>(Cuu_data.data());
        const auto Ouu = Kokkos::View<double[6][6]>(Ouu_data.data());
        const auto Puu = Kokkos::View<double[6][6]>(Puu_data.data());
        const auto Quu = Kokkos::View<double[6][6]>(Quu_data.data());

        KokkosBatched::SerialCopy<>::invoke(
            Kokkos::subview(qp_Cstar, i_elem, i_qp, Kokkos::ALL, Kokkos::ALL), Cstar
        );

        beams::InterpolateToQuadraturePointForStiffness(
            qp_jacobian(i_qp), Kokkos::subview(shape_interp, Kokkos::ALL, i_qp),
            Kokkos::subview(shape_deriv, Kokkos::ALL, i_qp), node_u, u, r, u_prime, r_prime
        );
        QuaternionCompose(r, r0, xr);

        masses::RotateSectionMatrix(xr, Cstar, Cuu);

        beams::CalculateStrain(x0_prime, u_prime, r, r_prime, strain);
        beams::CalculateTemporaryVariables(x0_prime, u_prime, x0pupSS);
        beams::CalculateForceFC(Cuu, strain, FC);
        beams::CalculateForceFD(x0pupSS, FC, FD);

        VecTilde(Kokkos::subview(FC, Kokkos::make_pair(0, 3)), N_tilde);
        VecTilde(Kokkos::subview(FC, Kokkos::make_pair(3, 6)), M_tilde);

        beams::CalculateOuu(Cuu, x0pupSS, M_tilde, N_tilde, Ouu);
        beams::CalculatePuu(Cuu, x0pupSS, N_tilde, Puu);
        beams::CalculateQuu(Cuu, x0pupSS, N_tilde, Quu);

        KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose, 1>::invoke(
            FC, Kokkos::subview(qp_Fc, i_qp, Kokkos::ALL)
        );
        KokkosBatched::SerialCopy<KokkosBatched::Trans::NoTranspose, 1>::invoke(
            FD, Kokkos::subview(qp_Fd, i_qp, Kokkos::ALL)
        );

        KokkosBatched::SerialCopy<>::invoke(
            Cuu, Kokkos::subview(qp_Cuu, i_qp, Kokkos::ALL, Kokkos::ALL)
        );
        KokkosBatched::SerialCopy<>::invoke(
            Ouu, Kokkos::subview(qp_Ouu, i_qp, Kokkos::ALL, Kokkos::ALL)
        );
        KokkosBatched::SerialCopy<>::invoke(
            Puu, Kokkos::subview(qp_Puu, i_qp, Kokkos::ALL, Kokkos::ALL)
        );
        KokkosBatched::SerialCopy<>::invoke(
            Quu, Kokkos::subview(qp_Quu, i_qp, Kokkos::ALL, Kokkos::ALL)
        );
    }
};

}  // namespace openturbine::beams
