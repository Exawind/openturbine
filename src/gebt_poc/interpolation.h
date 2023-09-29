#pragma once

#include <limits>

#include "src/gebt_poc/point.h"

namespace openturbine::gebt_poc {

/// Maximum number of iterations allowed for Newton's method
static constexpr size_t kMaxIterations{1000};

/// Tolerance for Newton's method to machine precision
static constexpr double kConvergenceTolerance{std::numeric_limits<double>::epsilon()};

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
   @details  Uses the recurrence relation for Legendre polynomials provided in
             Eq. B.1.15 (Page 446) of "High-Order Methods for Incompressible Fluid Flow"
             by Deville et al. 2002
             Ref: https://doi.org/10.1017/CBO9780511546792
 * @param  n: Order of the Legendre polynomial
 * @param  x: Point at which the Legendre polynomial is to be evaluated
 */
double LegendrePolynomial(const size_t n, const double x);

/*!
 * @brief  Evaluates the first derivative of Legendre polynomial of order n at point x recursively
 * @details  Uses the recurrence relation for Legendre polynomials provided in
             Eq. B.1.20 (Page 446) of "High-Order Methods for Incompressible Fluid Flow"
             by Deville et al. 2002
             Ref: https://doi.org/10.1017/CBO9780511546792
 * @param  n: Order of the Legendre polynomial
 * @param  x: Point at which the derivative of the Legendre polynomial is to be evaluated
 */
double LegendrePolynomialDerivative(const size_t n, const double x);

/*!
 * @brief  Determines the (n+1) Gauss-Lobatto-Legendre points required for nodal locations
 *         using polynomial shape/interpolation functions of order n
 * @details  Uses the Newton's method to find the roots of the Legendre polynomial which are
             the Gauss-Lobatto-Legendre points
 * @param  order: Order of the polynomial shape/interpolation functions
 * @return  Vector of Gauss-Lobatto-Legendre points with size n+1 for n+1 nodes
 */
std::vector<double> GenerateGLLPoints(const size_t order);

/*!
 * @brief Calculates the Lagrangian interpolation functions for order n at a given point x
   @details  Uses the relationship based on GLL points, Legendre polynomials and its
             derivative provided in Eq. 2.4.3 (Page 63) of "High-Order Methods for Incompressible
             Fluid Flow" by Deville et al. 2002
             Ref: https://doi.org/10.1017/CBO9780511546792
 * @param n: Order of the Legendre polynomial
 * @param x: Point at which the Lagrangian interpolation function is to be evaluated
 * @return  Vector of Lagrangian interpolation functions with size n+1 for n+1 nodes
 */
std::vector<double> LagrangePolynomial(const size_t n, const double x);

/*!
 * @brief Calculates the derivative of the Lagrangian interpolation functions for order n at a
          given point x
 * @details  Uses the relationship based on GLL points, Legendre polynomials and its
             derivative provided in Eq. 2.4.9 (Page 64) of "High-Order Methods for Incompressible
             Fluid Flow" by Deville et al. 2002
             Ref: https://doi.org/10.1017/CBO9780511546792
             And the generel formula for the derivative of a Lagrangian interpolation function
             provided here: https://en.wikipedia.org/wiki/Lagrange_polynomial#Derivatives
 * @param n: Order of the Legendre polynomial
 * @param x: Point at which the derivative of the Lagrangian interpolation function is to be
 *           evaluated
 * @return  Vector of derivative of Lagrangian interpolation functions with size n+1 for n+1 nodes
 */
std::vector<double> LagrangePolynomialDerivative(const size_t n, const double x);

}  // namespace openturbine::gebt_poc
