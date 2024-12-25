#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/math/vector_operations.hpp"

namespace openturbine::springs {

/**
 * @brief Functor to calculate force coefficients for spring elements
 */
struct CalculateForceCoefficients {
    size_t i_elem;                             //< Element index
    Kokkos::View<double*>::const_type k_;      //< Spring stiffness
    Kokkos::View<double*>::const_type l_ref_;  //< Reference length
    Kokkos::View<double*>::const_type l_;      //< Current length
    Kokkos::View<double*> c1_;                 //< Force coefficient 1
    Kokkos::View<double*> c2_;                 //< Force coefficient 2

    KOKKOS_FUNCTION
    void operator()() const {
        // c1 = k * (l_ref/l - 1)
        c1_(i_elem) = k_(i_elem) * (l_ref_(i_elem) / l_(i_elem) - 1.);
        // c2 = k * l_ref/(l^3)
        c2_(i_elem) = k_(i_elem) * l_ref_(i_elem) / (l_(i_elem) * l_(i_elem) * l_(i_elem));
    }
};

}  // namespace openturbine::springs
