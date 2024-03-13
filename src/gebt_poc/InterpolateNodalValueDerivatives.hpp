#pragma once

#include <vector>

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/gebt_poc/types.hpp"

namespace openturbine::gebt_poc {

inline void InterpolateNodalValueDerivatives(
    View2D::const_type nodal_values, std::vector<double> interpolation_function, double jacobian,
    View1D interpolated_values
) {
    if (jacobian == 0.) {
        throw std::invalid_argument("jacobian must be nonzero");
    }
    const auto n_nodes = nodal_values.extent(0);
    KokkosBlas::fill(interpolated_values, 0.);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        KokkosBlas::axpy(
            interpolation_function[i], Kokkos::subview(nodal_values, i, Kokkos::ALL),
            interpolated_values
        );
    }
    KokkosBlas::scal(interpolated_values, 1. / jacobian, interpolated_values);
}

KOKKOS_INLINE_FUNCTION
auto InterpolateNodalValueDerivatives(
    const Kokkos::TeamPolicy<>::member_type& member, View2D::const_type nodal_values,
    View1D::const_type interpolation_function, double jacobian
) {
    using scratch_space = Kokkos::DefaultExecutionSpace::scratch_memory_space;
    using unmanaged_memory = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using ScratchView1D = Kokkos::View<double*, scratch_space, unmanaged_memory>;
    auto interpolated_values = ScratchView1D(member.team_scratch(0), nodal_values.extent(1));
    const auto n_nodes = nodal_values.extent(0);
    const auto n_values = interpolated_values.extent(0);
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, n_values), [&](std::size_t i) {
        interpolated_values(i) = 0.;
    });
    member.team_barrier();
    Kokkos::parallel_for(
        Kokkos::ThreadVectorMDRange(member, n_nodes, n_values),
        [&](std::size_t i, std::size_t j) {
            auto result = nodal_values(i, j) * interpolation_function(i);
            Kokkos::atomic_add(&interpolated_values(j), result);
        }
    );
    member.team_barrier();
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, n_values), [&](std::size_t i) {
        interpolated_values(i) /= jacobian;
    });
    member.team_barrier();
    return interpolated_values;
}

}  // namespace openturbine::gebt_poc