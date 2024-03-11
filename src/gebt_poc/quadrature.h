#pragma once

#include <array>

#include "src/gebt_poc/types.hpp"

namespace openturbine::gebt_poc {

struct Quadrature {
  View1D points;
  View1D weights;
};

inline Quadrature CreateGaussLegendreQuadrature(int order) {
    if(order != 7) throw;
    static constexpr auto point_data = std::array{-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0., 0.4058451513773972, 0.7415311855993945, 0.9491079123427585};
    static constexpr auto weight_data = std::array{0.1294849661688697, 0.2797053914892766, 0.3818300505051189, 0.4179591836734694, 0.3818300505051189, 0.2797053914892766, 0.1294849661688697};

    auto points = View1D("points", order);
    auto weights = View1D("weights", order);

    using UnmanagedView1D = Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
    auto host_points = UnmanagedView1D(point_data.data(), point_data.size());
    auto host_weights = UnmanagedView1D(weight_data.data(), weight_data.size());
    Kokkos::deep_copy(points, host_points);
    Kokkos::deep_copy(weights, host_weights);

    return {points, weights};
}

}  // namespace openturbine::gebt_poc
