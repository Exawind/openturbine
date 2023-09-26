#pragma once

#include "src/gebt_poc/point.h"

namespace openturbine::gebt_poc {

/// Find the nearest neighbor of a point from a list
Point FindNearestNeighbor(const std::vector<Point>&, const Point&);

/*!
 * @brief  Find the k nearest neighbors of a point from a list
 * @param  k: Number of nearest neighbors to find
 */
std::vector<Point> FindkNearestNeighbors(const std::vector<Point>&, const Point&, const size_t k);

/*!
 * @brief  Perform linear interpolation between two matrices with the same dimensions
 * @param  alpha: Normalized distance of the interpolation point from the first matrix
 */
Kokkos::View<double**> LinearlyInterpolateMatrices(
    const Kokkos::View<double**>, const Kokkos::View<double**>, const double alpha
);

}  // namespace openturbine::gebt_poc
