#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename DeviceType>
struct ContributeSpringsToVector {
    typename Kokkos::View<size_t* [2][3], DeviceType>::const_type element_freedom_table;
    typename Kokkos::View<double* [2][3], DeviceType>::const_type elements;
    Kokkos::View<double* [1], Kokkos::LayoutLeft, DeviceType> vector;

    KOKKOS_FUNCTION
    void operator()(size_t element) const {
        constexpr auto force_atomic =
            !std::is_same_v<typename DeviceType::execution_space, Kokkos::Serial>;
        for (auto component = 0U; component < element_freedom_table.extent(2); ++component) {
            if constexpr (force_atomic) {
                Kokkos::atomic_add(
                    &vector(element_freedom_table(element, 0, component), 0),
                    elements(element, 0, component)
                );
                Kokkos::atomic_add(
                    &vector(element_freedom_table(element, 1, component), 0),
                    elements(element, 1, component)
                );
            } else {
                vector(element_freedom_table(element, 0, component), 0) +=
                    elements(element, 0, component);
                vector(element_freedom_table(element, 1, component), 0) +=
                    elements(element, 1, component);
            }
        }
    };
};

}  // namespace openturbine
