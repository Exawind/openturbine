#pragma once

#include <vector>

namespace openturbine {

/**
 * @brief Computes weights for linear interpolation
 *
 * @param x Evaluation point
 * @param xs Interpolation nodes (sorted)
 * @return weights Weights for linear interpolation (same size as xs)
 */
inline void LinearInterpWeights(
    double x, const std::vector<double>& xs, std::vector<double>& weights
) {
    const auto n = xs.size();
    weights.assign(n, 0.);

    const auto lower = std::lower_bound(xs.begin(), xs.end(), x);
    // If x is less than the first node, first weight -> 1 and our work is done
    if (lower == xs.begin()) {
        weights.front() = 1.;
        return;
    }
    // If x is greater than the last node, last weight -> 1 and our work is done
    if (lower == xs.end()) {
        weights.back() = 1.;
        return;
    }
    // x is between two nodes, so compute weights for closest nodes
    const auto index = static_cast<unsigned>(std::distance(xs.begin(), lower));
    const auto lower_loc = xs[index - 1];
    const auto upper_loc = xs[index];
    const auto weight_upper = (x - lower_loc) / (upper_loc - lower_loc);
    weights[index - 1] = 1. - weight_upper;
    weights[index] = weight_upper;
}

/**
 * @brief Computes weights for Lagrange polynomial interpolation
 *
 * @param x Evaluation point
 * @param xs Interpolation nodes (sorted)
 * @return weights Weights for Lagrange polynomial interpolation (same size as xs)
 */
inline void LagrangePolynomialInterpWeights(
    double x, const std::vector<double>& xs, std::vector<double>& weights
) {
    const auto n = xs.size();
    weights.assign(n, 1.);

    // Pre-compute (x - xs[m]) terms to avoid repeated calculations
    std::vector<double> x_minus_xm(n);
    for (size_t m = 0; m < n; ++m) {
        x_minus_xm[m] = x - xs[m];
    }

    for (size_t j = 0; j < n; ++j) {
        for (size_t m = 0; m < n; ++m) {
            if (j != m) {
                weights[j] *= x_minus_xm[m] / (xs[j] - xs[m]);
            }
        }
    }
}

/**
 * @brief Computes weights for Lagrange polynomial derivative interpolation
 *
 * @param x Evaluation point
 * @param xs Interpolation nodes (sorted)
 * @return weights Weights for Lagrange polynomial derivative interpolation (same size as xs)
 */
inline void LagrangePolynomialDerivWeights(
    double x, const std::vector<double>& xs, std::vector<double>& weights
) {
    const auto n = xs.size();
    weights.assign(n, 0.);

    // Pre-compute (x - xs[k]) terms to avoid repeated calculations
    std::vector<double> x_minus_xk(n);
    for (size_t k = 0; k < n; ++k) {
        x_minus_xk[k] = x - xs[k];
    }

    // Pre-compute denominators (xs[i] - xs[k]) to avoid repeated calculations
    std::vector<std::vector<double>> denom(n, std::vector<double>(n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < n; ++k) {
            if (k != i) {
                denom[i][k] = xs[i] - xs[k];
            }
        }
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (j != i) {
                double prod = 1.;
                for (size_t k = 0; k < n; ++k) {
                    if (k != i && k != j) {
                        prod *= x_minus_xk[k] / denom[i][k];
                    }
                }
                weights[i] += prod / denom[i][j];
            }
        }
    }
}

/**
 * @brief Calculates the value of Legendre polynomial of order n at point x
 * @details Uses the recurrence relation for Legendre polynomials:
 *          P_n(x) = ((2n-1)xP_{n-1}(x) - (n-1)P_{n-2}(x))/n
 *          Reference: Deville et al. (2002) "High-Order Methods for Incompressible Fluid Flow"
 *          DOI: https://doi.org/10.1017/CBO9780511546792, Eq. B.1.15, p.446
 *
 * @param n Order of the Legendre polynomial (n >= 0)
 * @param x Point at which to evaluate the polynomial, typically in [-1,1]
 * @return Value of the nth order Legendre polynomial at x
 */
inline double LegendrePolynomial(const size_t n, const double x) {
    // Base cases
    if (n == 0) {
        return 1.;
    }
    if (n == 1) {
        return x;
    }

    // Compute the nth Legendre polynomial iteratively
    double p_n_minus_2{1.};  // P_{n-2}(x)
    double p_n_minus_1{x};   // P_{n-1}(x)
    double p_n{0.};          // P_n(x)
    for (size_t i = 2; i <= n; ++i) {
        const auto i_double = static_cast<double>(i);
        p_n = ((2. * i_double - 1.) * x * p_n_minus_1 - (i_double - 1.) * p_n_minus_2) / i_double;
        p_n_minus_2 = p_n_minus_1;
        p_n_minus_1 = p_n;
    }
    return p_n;
}

}  // namespace openturbine
