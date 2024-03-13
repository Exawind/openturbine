#pragma once

#include <iostream>
#include <limits>

#include "src/gebt_poc/point.h"
#include "src/gebt_poc/types.hpp"

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
View2D LinearlyInterpolateMatrices(View2D::const_type, View2D::const_type, double alpha);

/*!
 * @brief  Calculates the value of Legendre polynomial of order n at point x recursively
   @details  Uses the recurrence relation for Legendre polynomials provided in
             Eq. B.1.15 (Page 446) of "High-Order Methods for Incompressible Fluid Flow"
             by Deville et al. 2002
             Ref: https://doi.org/10.1017/CBO9780511546792
 * @param  n: Order of the Legendre polynomial
 * @param  x: Point at which the Legendre polynomial is to be evaluated
 */
KOKKOS_INLINE_FUNCTION
static constexpr double LegendrePolynomial(const size_t n, const double x) {
    if (n == 0) {
        return 1.;
    }
    if (n == 1) {
        return x;
    }

    auto n_double = static_cast<double>(n);
    return (
        ((2. * n_double - 1.) * x * LegendrePolynomial(n - 1, x) -
         (n_double - 1.) * LegendrePolynomial(n - 2, x)) /
        n_double
    );
}

/*!
 * @brief  Evaluates the first derivative of Legendre polynomial of order n at point x recursively
 * @details  Uses the recurrence relation for Legendre polynomials provided in
             Eq. B.1.20 (Page 446) of "High-Order Methods for Incompressible Fluid Flow"
             by Deville et al. 2002
             Ref: https://doi.org/10.1017/CBO9780511546792
 * @param  n: Order of the Legendre polynomial
 * @param  x: Point at which the derivative of the Legendre polynomial is to be evaluated
 */
KOKKOS_INLINE_FUNCTION
static constexpr double LegendrePolynomialDerivative(const size_t n, const double x) {
    if (n == 0) {
        return 0.;
    }
    if (n == 1) {
        return 1.;
    }
    if (n == 2) {
        return (3. * x);
    }

    auto n_double = static_cast<double>(n);
    return (
        (2. * n_double - 1.) * LegendrePolynomial(n - 1, x) + LegendrePolynomialDerivative(n - 2, x)
    );
}

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
 * @param n: Order of the Lagrange polynomial
 * @param x: Point at which the Lagrangian interpolation function is to be evaluated
 * @return  Vector of Lagrangian interpolation functions with size n+1 for n+1 nodes
 */
std::vector<double> LagrangePolynomial(const size_t n, const double x);

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
std::vector<double> LagrangePolynomialDerivative(const size_t n, const double x);

KOKKOS_INLINE_FUNCTION
auto GenerateGLLPoints(const Kokkos::TeamPolicy<>::member_type& member, const size_t order) {
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using unmanaged_memory = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using ScratchView1D = Kokkos::View<double*, scratch_space, unmanaged_memory>;
    const auto n_nodes = order + 1;
    auto gll_points = ScratchView1D(member.team_scratch(0), n_nodes);
    Kokkos::single(Kokkos::PerTeam(member), [&]() {
        gll_points(0) = -1.;
        gll_points(order) = 1.;
    });

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, 1, n_nodes), [&](std::size_t i) {
        auto x_it = -std::cos(static_cast<double>(i) * M_PI / order);

        for (std::size_t j = 0; j < kMaxIterations; ++j) {
            const auto x_old = x_it;

            auto legendre_poly = ScratchView1D(member.thread_scratch(0), n_nodes);
            for (std::size_t k = 0; k < n_nodes; ++k) {
                legendre_poly(k) = LegendrePolynomial(k, x_it);
            }

            x_it -= (x_it * legendre_poly[n_nodes - 1] - legendre_poly[n_nodes - 2]) /
                    (n_nodes * legendre_poly[n_nodes - 1]);

            if (std::abs(x_it - x_old) <= kConvergenceTolerance) {
                break;
            }
        }

        gll_points(i) = x_it;
    });

    return gll_points;
}

KOKKOS_INLINE_FUNCTION
auto ComputeLagrangePolynomials(
    const Kokkos::TeamPolicy<>::member_type& member, std::size_t n, View1D::const_type points
) {
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using unmanaged_memory = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using ScratchView2D = Kokkos::View<double**, scratch_space, unmanaged_memory>;
    auto gll_points = GenerateGLLPoints(member, n);
    auto gll_point_v = GenerateGLLPoints(n);
    auto poly = ScratchView2D(member.team_scratch(0), points.extent(0), n + 1);
    member.team_barrier();

    Kokkos::parallel_for(
        Kokkos::ThreadVectorMDRange(member, points.extent(0), n + 1),
        [&](std::size_t i, std::size_t j) {
            const auto x = points(i);
            const auto gll = gll_points(j);
            if (std::abs(x - gll) < 1.e-15) {
                poly(i, j) = 1.;
            } else {
                poly(i, j) =
                    ((-1. / (n * (n + 1))) * ((1 - x * x) / (x - gll)) *
                     (LegendrePolynomialDerivative(n, x) / LegendrePolynomial(n, gll)));
            }
        }
    );

    return poly;
}

KOKKOS_INLINE_FUNCTION
auto ComputeLagrangePolynomialDerivatives(
    const Kokkos::TeamPolicy<>::member_type& member, std::size_t n, View1D::const_type points
) {
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using unmanaged_memory = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using ScratchView2D = Kokkos::View<double**, scratch_space, unmanaged_memory>;
    auto gll_points = GenerateGLLPoints(member, n);
    auto derivative = ScratchView2D(member.team_scratch(0), points.extent(0), n + 1);
    member.team_barrier();

    const auto n_double = static_cast<double>(n);
    Kokkos::parallel_for(
        Kokkos::ThreadVectorMDRange(member, points.extent(0), n + 1),
        [&](std::size_t i, std::size_t j) {
            const auto x = points(i);
            const auto gll = gll_points(j);

            if (std::abs(x - gll) < 1.e-15) {
                if (j == 0) {
                    derivative(i, j) = -n_double * (n_double + 1.) / 4.;
                } else if (j == n) {
                    derivative(i, j) = n_double * (n_double + 1.) / 4.;
                } else {
                    derivative(i, j) = 0.;
                }
            } else {
                derivative(i, j) = 0.;
                auto denominator = 1.;
                auto numerator = 1.;
                for (size_t k = 0; k < n + 1; ++k) {
                    if (k != j) {
                        denominator *= (gll - gll_points(k));
                    }
                    numerator = 1.;
                    for (size_t l = 0; l < n + 1; ++l) {
                        if (l != k && l != j && k != j) {
                            numerator *= (x - gll_points(l));
                        }
                        if (k == j) {
                            numerator = 0.;
                        }
                    }
                    derivative(i, j) += numerator;
                }
                derivative(i, j) /= denominator;
            }
        }
    );

    return derivative;
}

}  // namespace openturbine::gebt_poc
