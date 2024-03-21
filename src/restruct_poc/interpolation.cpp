#include "src/restruct_poc/interpolation.h"

namespace oturb {

void LagrangePolynomialInterpWeights(
    const double x, const std::vector<double>& xs, std::vector<double>& weights
) {
    // Get number of nodes
    auto n = xs.size();

    // Resize weights and fill with 1s
    weights.clear();
    weights.resize(n, 1.);

    // Calculate weights
    for (size_t j = 0; j < n; ++j) {
        for (size_t m = 0; m < n; ++m) {
            if (j != m) {
                weights[j] *= (x - xs[m]) / (xs[j] - xs[m]);
            }
        }
    }
}

void LagrangePolynomialDerivWeights(
    const double x, const std::vector<double>& xs, std::vector<double>& weights
) {
    // Get number of nodes
    auto n = xs.size();

    // Resize weights and fill with zeros
    weights.clear();
    weights.resize(n, 0.);

    // Calculate weights
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

}  // namespace oturb
