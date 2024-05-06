#pragma once

#include <Kokkos_Core.hpp>

#include "src/restruct_poc/types.hpp"

namespace openturbine {

template <typename Subview_NxN>
struct UpdateIterationMatrix {
    Subview_NxN St_12;
    View_NxN::const_type B;

    KOKKOS_FUNCTION
    void operator()(const int i, const int j) const { St_12(j, i) = B(i, j); }
};

}  // namespace openturbine
