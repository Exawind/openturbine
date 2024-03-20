#pragma once

#include <vector>

namespace oturb {

/// Maximum number of iterations allowed for Newton's method
static constexpr size_t kMaxIterations{1000};

/// Tolerance for Newton's method to machine precision
static constexpr double kConvergenceTolerance{std::numeric_limits<double>::epsilon()};

/*!
 * @brief Calculates the Lagrangian interpolation functions for order n at a given point x
   @details  Uses the relationship based on GLL points, Legendre polynomials and its
             derivative provided in Eq. 2.4.3 (Page 63) of "High-Order Methods for Incompressible
             Fluid Flow" by Deville et al. 2002
             Ref: https://doi.org/10.1017/CBO9780511546792
 * @param n: Order of the Lagrange polynomial
 * @param x: Point at which the Lagrangian interpolation function is to be evaluated
 * @return  Vector of Lagrangian interpolation functions with size n+1 for n+1 nodes
 */
void LagrangePolynomial(const size_t n, const double x, std::vector<double>& weights);

/*!
 * @brief Calculates the derivative of the Lagrangian interpolation functions for order n at a
          given point x
 * @details  Uses the relationship provided in Eq. 2.4.9 (Page 64) of "High-Order Methods for
             Incompressible Fluid Flow" by Deville et al. 2002
             Ref: https://doi.org/10.1017/CBO9780511546792
             And the general formula for the derivative of a Lagrangian interpolation function
             provided here: https://en.wikipedia.org/wiki/Lagrange_polynomial#Derivatives
 * @param n: Order of the Lagrange polynomial
 * @param x: Point at which the derivative of the Lagrangian interpolation function is to be
 *           evaluated
 * @return  Vector of derivative of Lagrangian interpolation functions with size n+1 for n+1 nodes
 */
void LagrangePolynomialDerivative(const size_t n, const double x, std::vector<double>& weights);

void LagrangePolynomialInterpWeights(
    const double x, const std::vector<double>& xs, std::vector<double>& weights
);
void LagrangePolynomialDerivWeights(
    const double x, const std::vector<double>& xs, std::vector<double>& weights
);

}  // namespace oturb
