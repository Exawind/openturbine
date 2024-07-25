#pragma once

#include <vector>

namespace openturbine {

inline void LinearInterpWeights(
    double x, const std::vector<double>& xs, std::vector<double>& weights
) {
    const auto n = xs.size();

    weights.clear();
    weights.resize(n, 0.);

    const auto lower = std::lower_bound(xs.begin(), xs.end(), x);

    if (lower == xs.begin()) {
        weights.front() = 1.0;
    } else if (lower == xs.end()) {
        weights.back() = 1.0;
    } else {
        auto index = static_cast<unsigned>(std::distance(xs.begin(), lower));
        auto lower_loc = xs[index - 1];
        auto upper_loc = xs[index];
        auto weight_upper = (x - lower_loc) / (upper_loc - lower_loc);
        weights[index - 1] = 1.0 - weight_upper;
        weights[index] = weight_upper;
    }
}

inline void LagrangePolynomialInterpWeights(
    double x, const std::vector<double>& xs, std::vector<double>& weights
) {
    const auto n = xs.size();

    weights.clear();
    weights.resize(n, 1.);

    for (size_t j = 0; j < n; ++j) {
        for (size_t m = 0; m < n; ++m) {
            if (j != m) {
                weights[j] *= (x - xs[m]) / (xs[j] - xs[m]);
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
