#pragma once

#include <Kokkos_Core.hpp>

#include "types.hpp"

namespace openturbine {

template <typename DeviceType>
struct ConditionR {
    double conditioner;
    Kokkos::View<double* [1], Kokkos::LayoutLeft, DeviceType> R;

    KOKKOS_FUNCTION
    void operator()(int i) const { R(i, 0) *= conditioner; }
};

template <typename DeviceType>
struct UnconditionSolution {
    size_t num_system_dofs;
    double conditioner;
    Kokkos::View<double* [1], Kokkos::LayoutLeft, DeviceType> x;

    KOKKOS_FUNCTION
    void operator()(size_t i) const { x(i + num_system_dofs, 0) /= conditioner; }
};

}  // namespace openturbine
