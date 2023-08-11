#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine::util {

// Used to store a 1D Kokkos View of doubles on the host
using HostView1D = Kokkos::View<double*, Kokkos::HostSpace>;

// Used to store a 2D Kokkos View of doubles on the host
using HostView2D = Kokkos::View<double**, Kokkos::HostSpace>;

}  // namespace openturbine::util
