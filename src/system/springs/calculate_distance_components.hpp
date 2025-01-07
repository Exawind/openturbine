#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/math/vector_operations.hpp"

namespace openturbine::springs {

/**
 * @brief Functor to calculate relative distance vector between spring element nodes
 */
struct CalculateDistanceComponents {
    size_t i_elem;                              //< Element index
    Kokkos::View<double* [3]>::const_type x0_;  //< Initial distance vector between nodes
    Kokkos::View<double* [3]>::const_type u1_;  //< Displacement vector of node 1
    Kokkos::View<double* [3]>::const_type u2_;  //< Displacement vector of node 2
    Kokkos::View<double* [3]> r_;               //< Relative distance vector between the two nodes

    KOKKOS_FUNCTION
    void operator()() const {
        auto x0 = Kokkos::subview(x0_, i_elem, Kokkos::ALL);
        auto u1 = Kokkos::subview(u1_, i_elem, Kokkos::ALL);
        auto u2 = Kokkos::subview(u2_, i_elem, Kokkos::ALL);
        auto r = Kokkos::subview(r_, i_elem, Kokkos::ALL);

        for (int i = 0; i < 3; ++i) {
            r(i) = x0(i) + u2(i) - u1(i);
        }
    }
};

}  // namespace openturbine::springs
