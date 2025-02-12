#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "math/vector_operations.hpp"

namespace openturbine::springs {

KOKKOS_FUNCTION
inline double CalculateForceCoefficient1(double k, double l_ref, double l) {
    return k * (l_ref / l - 1.);
}

KOKKOS_FUNCTION
inline double CalculateForceCoefficient2(double k, double l_ref, double l) {
    return k * l_ref / (l * l * l);
}

}  // namespace openturbine::springs
