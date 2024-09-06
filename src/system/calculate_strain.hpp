#pragma once

#include <KokkosBatched_Gemv_Decl.hpp>
#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/math/quaternion_operations.hpp"

namespace openturbine {

struct CalculateStrain {
    using NoTranspose = KokkosBlas::Trans::NoTranspose;
    using Default = KokkosBlas::Algo::Gemv::Default;
    using Gemv = KokkosBlas::SerialGemv<NoTranspose, Default>;
    size_t i_elem;
    Kokkos::View<double** [3]>::const_type qp_x0_prime_;
    Kokkos::View<double** [3]>::const_type qp_u_prime_;
    Kokkos::View<double** [4]>::const_type qp_r_;
    Kokkos::View<double** [4]>::const_type qp_r_prime_;
    Kokkos::View<double** [6]> qp_strain_;

    KOKKOS_FUNCTION
    void operator()(const int i_qp) const {
        auto x0_prime_data = Kokkos::Array<double, 3>{
            qp_x0_prime_(i_elem, i_qp, 0), qp_x0_prime_(i_elem, i_qp, 1),
            qp_x0_prime_(i_elem, i_qp, 2)};
        auto x0_prime = Kokkos::View<double[3]>(x0_prime_data.data());
        auto u_prime_data = Kokkos::Array<double, 3>{
            qp_u_prime_(i_elem, i_qp, 0), qp_u_prime_(i_elem, i_qp, 1),
            qp_u_prime_(i_elem, i_qp, 2)};
        auto u_prime = Kokkos::View<double[3]>(u_prime_data.data());
        auto R_data = Kokkos::Array<double, 4>{
            qp_r_(i_elem, i_qp, 0), qp_r_(i_elem, i_qp, 1), qp_r_(i_elem, i_qp, 2),
            qp_r_(i_elem, i_qp, 3)};
        auto R = Kokkos::View<double[4]>(R_data.data());

        auto R_x0_prime_data = Kokkos::Array<double, 3>{};
        auto R_x0_prime = Kokkos::View<double[3]>(R_x0_prime_data.data());

        RotateVectorByQuaternion(R, x0_prime, R_x0_prime);
        KokkosBlas::serial_axpy(-1., u_prime, R_x0_prime);
        KokkosBlas::serial_axpy(-1., x0_prime, R_x0_prime);

        auto E_data = Kokkos::Array<double, 12>{};
        auto E = Kokkos::View<double[3][4]>(E_data.data());
        QuaternionDerivative(R, E);
        auto R_prime_data = Kokkos::Array<double, 4>{
            qp_r_prime_(i_elem, i_qp, 0), qp_r_prime_(i_elem, i_qp, 1), qp_r_prime_(i_elem, i_qp, 2),
            qp_r_prime_(i_elem, i_qp, 3)};
        auto R_prime = Kokkos::View<double[4]>(R_prime_data.data());
        auto e2_data = Kokkos::Array<double, 3>{};
        auto e2 = Kokkos::View<double[4]>{e2_data.data()};
        Gemv::invoke(2., E, R_prime, 0., e2);

        qp_strain_(i_elem, i_qp, 0) = -R_x0_prime(0);
        qp_strain_(i_elem, i_qp, 1) = -R_x0_prime(1);
        qp_strain_(i_elem, i_qp, 2) = -R_x0_prime(2);
        qp_strain_(i_elem, i_qp, 3) = e2(0);
        qp_strain_(i_elem, i_qp, 4) = e2(1);
        qp_strain_(i_elem, i_qp, 5) = e2(2);
    }
};

}  // namespace openturbine
