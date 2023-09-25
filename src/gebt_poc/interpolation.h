#pragma once

#include "src/gebt_poc/point.h"

namespace openturbine::gebt_poc {

///< Maximum number of iterations allowed for Newton's method
static constexpr size_t kMaxIterations{1000};

///< Tolerance for Newton's method to machine precision
static constexpr double kTolerance{1e-15};

/// Finds the nearest neighbor of a point from a list
Point FindNearestNeighbor(const std::vector<Point>&, const Point&);

/*!
 * @brief  Finds the k nearest neighbors of a point from a list
 * @param  k: Number of nearest neighbors to find
 */
std::vector<Point> FindkNearestNeighbors(const std::vector<Point>&, const Point&, const size_t k);

/*!
 * @brief  Performs linear interpolation between two matrices with the same dimensions
 * @param  alpha: Normalized distance of the interpolation point from the first matrix
 */
Kokkos::View<double**> LinearlyInterpolateMatrices(
    const Kokkos::View<double**>, const Kokkos::View<double**>, const double alpha
);

/*!
 * @brief  Calculates the value of Legendre polynomial of order n at point x recursively
 * @param  n: Order of the Legendre polynomial
 * @param  x: Point at which the Legendre polynomial is to be evaluated
 */
double LegendrePolynomial(const size_t n, const double x);

/*!
 * @brief  Determines the (n+1) Gauss-Lobatto-Legendre points required for nodal locations
 *         using polynomial shape/interpolation functions of order n
 * @param  order: Order of the polynomial shape/interpolation functions
 */
std::vector<double> GenerateGLLPoints(const size_t order);

/*!
 * @brief  Evaluates the first derivative of Legendre polynomial of order n at point x recursively
 * @param  n: Order of the Legendre polynomial
 * @param  x: Point at which the derivative of the Legendre polynomial is to be evaluated
 */
double LegendrePolynomialDerivative(const size_t n, const double x);

/*!
 * @brief Calculates the Lagrangian interpolation functions for order n at a given point x
 * @param n: Order of the Legendre polynomial
 * @param x: Point at which the Lagrangian interpolation function is to be evaluated
 */
std::vector<double> LagrangePolynomial(const size_t n, const double x);

}  // namespace openturbine::gebt_poc
