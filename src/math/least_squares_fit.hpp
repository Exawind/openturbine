#pragma once

#include <array>
#include <stdexcept>
#include <vector>

#include "src/beams/interpolation.hpp"

namespace openturbine {

// Step 1: Mapping Geometric Locations
/**
 * @brief Maps input geometric locations -> normalized domain using linear mapping
 *
 * @param geom_input_locations Input geometric locations (typically in domain [0, 1]),
 *                             sorted
 * @return std::vector<double> Mapped/normalized evaluation points in domain [-1, 1]
 */
std::vector<double> MapGeometricLocations(const std::vector<double>& geom_input_locations) {
    // Get first and last points of the input domain (assumed to be sorted)
    double domain_start = geom_input_locations.front();
    double domain_end = geom_input_locations.back();
    if (domain_end == domain_start) {
        throw std::invalid_argument(
            "Invalid geometric locations: domain start and end points are equal."
        );
    }

    // Map each point from domain -> [-1, 1]
    std::vector<double> mapped_locations(geom_input_locations.size());
    auto domain_span = domain_end - domain_start;
    for (size_t i = 0; i < geom_input_locations.size(); ++i) {
        mapped_locations[i] = 2. * (geom_input_locations[i] - domain_start) / domain_span - 1.;
    }
    return mapped_locations;
}

// Step 2: Shape Function Matrices
/**
 * @brief Computes shape function matrices ϕg at points ξg
 *
 * @param n Number of geometric points to fit (>=2)
 * @param p Number of points representing the polynomial of order p-1 (2 <= p <= n)
 * @param xi_g Evaluation points in [-1, 1]
 * @return Tuple containing shape function matrix and GLL points
 */
std::tuple<std::vector<std::vector<double>>, std::vector<double>> ShapeFunctionMatrices(
    size_t n, size_t p, const std::vector<double>& xi_g
) {
    // Compute GLL points which will act as the nodes for the shape functions
    auto gll_pts = GenerateGLLPoints(p - 1);

    // Compute weights for the shape functions at the nodes
    std::vector<double> weights(p, 0.);
    std::vector<std::vector<double>> phi_g(p, std::vector<double>(n, 0.));
    for (size_t j = 0; j < n; ++j) {
        LagrangePolynomialInterpWeights(xi_g[j], gll_pts, weights);
        for (size_t k = 0; k < p; ++k) {
            phi_g[k][j] = weights[k];
        }
    }

    return {phi_g, gll_pts};
}

}  // namespace openturbine
