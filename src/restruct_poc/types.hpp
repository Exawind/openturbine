#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

static constexpr int kVectorComponents = 3;
static constexpr int kLieAlgebraComponents = 6;
static constexpr int kLieGroupComponents = 7;

static constexpr double kTolerance = 1.e-16;

// Create some type aliases for Kokkos views to improve readability
// 1D views
using View_3 = Kokkos::View<double[3]>;
using View_Quaternion = Kokkos::View<double[4]>;
using View_N = Kokkos::View<double*>;
// 2D views
using View_3x3 = Kokkos::View<double[3][3]>;
using View_3x4 = Kokkos::View<double[3][4]>;
using View_6x6 = Kokkos::View<double[6][6]>;
using View_Nx3 = Kokkos::View<double* [3]>;
using View_Nx4 = Kokkos::View<double* [4]>;
using View_Nx6 = Kokkos::View<double* [6]>;
using View_Nx7 = Kokkos::View<double* [7]>;
using View_NxN = Kokkos::View<double**>;
// 3D views
using View_Nx3x3 = Kokkos::View<double* [3][3]>;
using View_Nx3x4 = Kokkos::View<double* [3][4]>;
using View_Nx6x6 = Kokkos::View<double* [6][6]>;

// Define some type aliases for 2D std::arrays to improve readability
using Array_6x6 = std::array<std::array<double, 6>, 6>;

// Atomic
using View_N_atomic = Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::Atomic>>;
using View_NxN_atomic = Kokkos::View<double**, Kokkos::MemoryTraits<Kokkos::Atomic>>;

}  // namespace openturbine
