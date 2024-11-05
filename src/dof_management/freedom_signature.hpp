#pragma once

#include <type_traits>

#include <Kokkos_Core.hpp>

namespace openturbine {

enum class FreedomSignature : std::uint8_t {
    AllComponents = 0b01111111,
    JustPosition = 0b01110000,
    JustRotation = 0b00001111,
    NoComponents = 0b00000000
};

KOKKOS_INLINE_FUNCTION
FreedomSignature operator|(FreedomSignature x, FreedomSignature y) {
    using T = std::underlying_type_t<FreedomSignature>;
    return static_cast<FreedomSignature>(static_cast<T>(x) | static_cast<T>(y));
}

KOKKOS_INLINE_FUNCTION
size_t count_active_dofs(FreedomSignature x) {
    using T = std::underlying_type_t<FreedomSignature>;
    auto count = 0ul;
    constexpr auto zero = T{0};
    constexpr auto one = T{1};
    for(auto value = static_cast<T>(x); value > zero; value = value >> 1) {
        count += value & one;
    }
    return count;
}

}
