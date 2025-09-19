#pragma once

#include <algorithm>
#include <array>
#include <ranges>
#include <span>
#include <vector>

namespace kynema::beams {

inline std::vector<std::array<double, 2>> CreateTrapezoidalQuadrature(std::span<const double> grid) {
    const auto n{grid.size()};
    const auto [grid_min, grid_max] = std::ranges::minmax(grid);
    const auto grid_range{grid_max - grid_min};
    auto quadrature = std::vector<std::array<double, 2>>{
        {-1., (grid[1] - grid[0]) / grid_range},
    };
    std::ranges::transform(
        std::views::iota(1U, n - 1), std::back_inserter(quadrature),
        [grid, gm = grid_min, grid_range](auto i) {
            return std::array{
                2. * (grid[i] - gm) / grid_range - 1., (grid[i + 1] - grid[i - 1]) / grid_range
            };
        }
    );
    quadrature.push_back({1., (grid[n - 1] - grid[n - 2]) / grid_range});
    return quadrature;
}

}  // namespace kynema::beams
