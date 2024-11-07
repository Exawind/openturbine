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

inline void LagrangePolynomialDerivWeights(
    double x, const std::vector<double>& xs, std::vector<double>& weights
) {
    const auto n = xs.size();

    weights.clear();
    weights.resize(n, 0.);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (j != i) {
                double prod = 1.0;
                for (size_t k = 0; k < n; ++k) {
                    if (k != i && k != j) {
                        prod *= (x - xs[k]) / (xs[i] - xs[k]);
                    }
                }
                weights[i] += prod / (xs[i] - xs[j]);
            }
        }
    }
}

}  // namespace openturbine
