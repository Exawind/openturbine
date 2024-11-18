#pragma once

#include <Kokkos_Core.hpp>

#include "src/math/vector_operations.hpp"

namespace openturbine {

/**
 * @brief Functor to calculate mass matrix components from mass matrix in inertial csys
 *
 * This struct serves as a function object to compute three key components from the provided mass
 * matrix:
 * - eta: The offset vector representing the distance between mass center and elastic axis
 * - rho: The 3x3 mass matrix for rotational terms
 * - eta_tilde: The skew-symmetric matrix derived from eta
 *
 * The calculations are performed for each quadrature point (i_qp) of a given element (i_elem)
 */
struct CalculateMassMatrixComponents {
    size_t i_elem;                                      //< Element index
    Kokkos::View<double** [6][6]>::const_type qp_Muu_;  //< Mass matrix in inertial csys
    Kokkos::View<double** [3]> eta_;                    //< Offset between mass center and
                                                        //< elastic axis
    Kokkos::View<double** [3][3]> rho_;                 //< Rotational part of mass matrix
    Kokkos::View<double** [3][3]> eta_tilde_;           //< Skew-symmetric matrix derived from eta

    KOKKOS_FUNCTION
    void operator()(int i_qp) const {
        auto Muu = Kokkos::subview(qp_Muu_, i_elem, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto eta = Kokkos::subview(eta_, i_elem, i_qp, Kokkos::ALL);
        auto rho = Kokkos::subview(rho_, i_elem, i_qp, Kokkos::ALL, Kokkos::ALL);
        auto eta_tilde = Kokkos::subview(eta_tilde_, i_elem, i_qp, Kokkos::ALL, Kokkos::ALL);

        // Compute eta components using off-diagonal terms in lower left 3x3 block of Muu
        eta(0) = eta(1) = eta(2) = 0.;
        if (const auto m = Muu(0, 0); m != 0.) {
            eta(0) = Muu(5, 1) / m;
            eta(1) = -Muu(5, 0) / m;
            eta(2) = Muu(4, 0) / m;
        }

        // Extract the rotational mass terms from the lower-right 3x3 block of Muu
        for (int i = 0; i < rho.extent_int(0); ++i) {
            for (int j = 0; j < rho.extent_int(1); ++j) {
                rho(i, j) = Muu(i + 3, j + 3);
            }
        }

        // Generate the skew-symmetric matrix eta_tilde from eta
        VecTilde(eta, eta_tilde);
    }
};

}  // namespace openturbine
