#pragma once

#include <Kokkos_Core.hpp>

namespace oturb {

static constexpr std::size_t LieAlgebraComponents = 6;
static constexpr std::size_t LieGroupComponents = 7;
static constexpr std::size_t VectorComponents = 3;

static constexpr double Tolerance = 1.e-16;

using LieAlgebraFieldView = Kokkos::View<double* [LieAlgebraComponents]>;
using LieGroupFieldView = Kokkos::View<double* [LieGroupComponents]>;
using VectorFieldView = Kokkos::View<double* [VectorComponents]>;

using View1D = Kokkos::View<double*>;
using View1D_Vector = Kokkos::View<double[VectorComponents]>;
using View1D_LieAlgebra = Kokkos::View<double[LieAlgebraComponents]>;
using View1D_LieGroup = Kokkos::View<double[LieGroupComponents]>;

using View_Rot = Kokkos::View<double[3][3]>;
using View_3x3 = Kokkos::View<double[3][3]>;
using View_3x4 = Kokkos::View<double[3][4]>;
using View_6x6 = Kokkos::View<double[6][6]>;

using View_N = Kokkos::View<double*>;
using View_3 = Kokkos::View<double[3]>;
using View_Quat = Kokkos::View<double[4]>;

using View_NxN = Kokkos::View<double**>;
using View_Nx3 = Kokkos::View<double* [3]>;
using View_Nx4 = Kokkos::View<double* [4]>;
using View_Nx6 = Kokkos::View<double* [6]>;
using View_Nx7 = Kokkos::View<double* [7]>;

using View_Nx3x4 = Kokkos::View<double* [3][4]>;
using View_Nx3x3 = Kokkos::View<double* [3][3]>;
using View_Nx6x6 = Kokkos::View<double* [6][6]>;

}  // namespace oturb