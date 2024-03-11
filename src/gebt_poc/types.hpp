#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

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

using View2D = Kokkos::View<double**>;
using View2D_3x3 = Kokkos::View<double[3][3]>;
using View2D_6x6 = Kokkos::View<double[6][6]>;

}  // namespace openturbine