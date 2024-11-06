#pragma once

#include <array>
#include <stdexcept>
#include <vector>

namespace openturbine {

// Step 1: Mapping Geometric Locations
/**
 * @brief Maps input geometric locations -> normalized domain using linear mapping
 *
 * @param geom_input_locations Input geometric locations in domain [0, 1], sorted
 * @return std::vector<double> Mapped/normalized evaluation points in domain [-1, 1]
 */
std::vector<double> MapGeometricLocations(const std::vector<double>& geom_input_locations) {
    // Get first and last points of the input domain assumed to be sorted
    double domain_start = geom_input_locations.front();
    double domain_end = geom_input_locations.back();
    if (domain_end == domain_start) {
        throw std::invalid_argument(
            "Invalid geometric locations: domain start and end points are equal."
        );
    }

    std::vector<double> mapped_locations;
    mapped_locations.reserve(geom_input_locations.size());
    // Map each point from [domain_start, domain_end] to [-1, 1]
    for (const auto& location : geom_input_locations) {
        double normalized_point = 2. * (location - domain_start) / (domain_end - domain_start) - 1.;
        mapped_locations.push_back(normalized_point);
    }
    return mapped_locations;
}

}  // namespace openturbine
