#pragma once

#include <algorithm>
#include <array>
#include <vector>

namespace openturbine::beams {

inline std::vector<std::array<double, 2>> CreateTrapezoidalQuadrature(const std::vector<double>& grid
) {
    const auto n{grid.size()};
    const auto [grid_min, grid_max] = std::minmax_element(begin(grid), end(grid));
    const auto grid_range{*grid_max - *grid_min};
    auto quadrature = std::vector<std::array<double, 2>>{
        {-1., (grid[1] - grid[0]) / grid_range},
    };
    for (auto i = 1U; i < n - 1; ++i) {
        quadrature.push_back(
            {2. * (grid[i] - *grid_min) / grid_range - 1., (grid[i + 1] - grid[i - 1]) / grid_range}
        );
    }
    quadrature.push_back({1., (grid[n - 1] - grid[n - 2]) / grid_range});
    return quadrature;
}

}  // namespace openturbine::beams
