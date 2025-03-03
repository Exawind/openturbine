#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "math/vector_operations.hpp"

namespace openturbine::springs {

KOKKOS_FUNCTION
inline double CalculateLength(const Kokkos::View<double[3]>::const_type& r) {
    return sqrt(r(0) * r(0) + r(1) * r(1) + r(2) * r(2));
}

}  // namespace openturbine::springs
