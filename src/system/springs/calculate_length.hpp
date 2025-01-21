#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "math/vector_operations.hpp"

namespace openturbine::springs {

/**
 * @brief Functor to calculate the current length of a spring element
 */
struct CalculateLength {
    size_t i_elem;                             //< Element index
    Kokkos::View<double* [3]>::const_type r_;  //< Relative distance vector between nodes
    Kokkos::View<double*> l_;                  //< Current length

    KOKKOS_FUNCTION
    void operator()() const {
        auto r = Kokkos::subview(r_, i_elem, Kokkos::ALL);
        l_(i_elem) = sqrt(r(0) * r(0) + r(1) * r(1) + r(2) * r(2));
    }
};

}  // namespace openturbine::springs
