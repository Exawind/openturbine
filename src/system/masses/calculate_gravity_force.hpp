#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/math/vector_operations.hpp"

namespace openturbine::masses {

/**
 * @brief Functor to calculate gravity forces in a beam/mass element
 *
 * This struct serves as a function object to compute gravity forces for beam and rigid body
 * elements.
 *
 * Gravity force vector, FG = {
 *     {FG_1} = {m * g}
 *     {FG_2} = {m * eta_tilde * g}
 * }
 *
 * The forces are computed for each quadrature point (i_qp) of a given element (i_elem).
 */
struct CalculateGravityForce {
    using NoTranspose = KokkosBlas::Trans::NoTranspose;
    using Default = KokkosBlas::Algo::Gemv::Default;
    using Gemv = KokkosBlas::SerialGemv<NoTranspose, Default>;
    size_t i_elem;
    Kokkos::View<double[3]>::const_type gravity;           //< Gravitational acceleration vector
    Kokkos::View<double* [6][6]>::const_type qp_Muu_;     //< Mass matrix in inertial csys
    Kokkos::View<double* [3][3]>::const_type eta_tilde_;  //< Skew-symmetric matrix derived from eta
    Kokkos::View<double* [6]> qp_FG_;  //< Gravity forces (computed in this functor)

    KOKKOS_FUNCTION
    void operator()() const {
        auto Muu = Kokkos::subview(qp_Muu_, i_elem, Kokkos::ALL, Kokkos::ALL);
        auto eta_tilde = Kokkos::subview(eta_tilde_, i_elem, Kokkos::ALL, Kokkos::ALL);
        auto FG = Kokkos::subview(qp_FG_, i_elem, Kokkos::ALL);
        // Compute FG_1 = m * {g}
        auto m = Muu(0, 0);
        for (int i = 0; i < 3; ++i) {
            FG(i) = m * gravity(i);
        }
        // Compute FG_2 = m * [eta_tilde] * {g}
        Gemv::invoke(
            1., eta_tilde, Kokkos::subview(FG, Kokkos::make_pair(0, 3)), 0.,
            Kokkos::subview(FG, Kokkos::make_pair(3, 6))
        );
    }
};

}  // namespace openturbine
