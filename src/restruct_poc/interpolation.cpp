#include "src/restruct_poc/interpolation.h"

namespace oturb {

double LegendrePolynomial(const size_t n, const double x) {
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

double LegendrePolynomialDerivative(const size_t n, const double x) {
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

std::vector<double> GenerateGLLPoints(const size_t order) {
    if (order < 1) {
        throw std::invalid_argument("Polynomial order must be greater than or equal to 1");
    }

    auto n_nodes = order + 1;  // number of nodes = order + 1
    std::vector<double> gll_points(n_nodes);
    gll_points[0] = -1.;     // left end point
    gll_points[order] = 1.;  // right end point

    for (size_t i = 1; i < n_nodes; ++i) {
        // Use the Chebyshev-Gauss-Lobatto nodes as the initial guess
        auto x_it = -std::cos(static_cast<double>(i) * M_PI / order);

        // Use Newton's method to find the roots of the Legendre polynomial
        for (size_t j = 0; j < kMaxIterations; ++j) {
            auto x_old = x_it;

            auto legendre_poly = std::vector<double>(n_nodes);
            for (size_t k = 0; k < n_nodes; ++k) {
                legendre_poly[k] = LegendrePolynomial(k, x_it);
            }

            x_it -= (x_it * legendre_poly[n_nodes - 1] - legendre_poly[n_nodes - 2]) /
                    (n_nodes * legendre_poly[n_nodes - 1]);

            if (std::abs(x_it - x_old) <= kConvergenceTolerance) {
                break;
            }
        }
        gll_points[i] = x_it;
    }

    return gll_points;
}

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
    weights.resize(n, 1.);

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

void LagrangePolynomial(const size_t n, const double x, std::vector<double>& weights) {
    auto gll_points = GenerateGLLPoints(n);  // n+1 GLL points for n+1 nodes
    weights.clear();
    weights.resize(n + 1, 1.);  // resize and fill

    for (size_t i = 0; i < n + 1; ++i) {
        weights[i] =
            ((-1. / (n * (n + 1))) * ((1 - x * x) / (x - gll_points[i])) *
             (LegendrePolynomialDerivative(n, x) / LegendrePolynomial(n, gll_points[i])));
    }
}

void LagrangePolynomialDerivative(const size_t n, const double x, std::vector<double>& weights) {
    auto gll_points = GenerateGLLPoints(n);  // n+1 GLL points for n+1 nodes
    weights.clear();
    weights.resize(n + 1, 0.);

    for (size_t i = 0; i < n + 1; ++i) {
        // Calculates derivative based on Legendre polynomial - works for only at GLL points
        // lagrange_poly_derivative[i] =
        //     (1. / (x - gll_points[i])) *
        //     (LegendrePolynomial(n, x) / LegendrePolynomial(n, gll_points[i]));

        // Calculates derivative based on general def. of Lagrange interpolants - works for GLL
        // points and any other point
        auto denominator = 1.;
        auto numerator = 1.;
        for (size_t j = 0; j < n + 1; ++j) {
            if (j != i) {
                denominator *= (gll_points[i] - gll_points[j]);
            }
            numerator = 1.;
            for (size_t k = 0; k < n + 1; ++k) {
                if (k != j && k != i && j != i) {
                    numerator *= (x - gll_points[k]);
                }
                if (j == i) {
                    numerator = 0.;
                }
            }
            weights[i] += numerator;
        }
        weights[i] /= denominator;
    }
}

}  // namespace oturb
