#pragma once
#include <vector>

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>

#include "src/gebt_poc/types.hpp"

namespace openturbine::gebt_poc {

inline void InterpolateNodalValues(
    View2D::const_type nodal_values, std::vector<double> interpolation_function,
    View1D interpolated_values
) {
    Kokkos::deep_copy(interpolated_values, 0.);
    const auto n_nodes = nodal_values.extent(0);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        KokkosBlas::axpy(
            interpolation_function[i], Kokkos::subview(nodal_values, i, Kokkos::ALL),
            interpolated_values
        );
    }

    // Normalize the rotation quaternion if it is not already normalized
    if (nodal_values.extent(1) == LieGroupComponents) {
        auto q = Kokkos::subview(interpolated_values, Kokkos::pair(3, 7));
        if (auto norm = KokkosBlas::nrm2(q); norm != 0. && norm != 1.) {
            KokkosBlas::scal(q, 1. / norm, q);
        }
    }
}

KOKKOS_INLINE_FUNCTION
auto InterpolateNodalValues(
    const Kokkos::TeamPolicy<>::member_type& member, View2D::const_type nodal_values,
    View1D::const_type interpolation_function
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
    // Normalize the rotation quaternion if it is not already normalized
    if (nodal_values.extent(1) == LieGroupComponents) {
        Kokkos::single(Kokkos::PerTeam(member), [&]() {
            const auto norm = std::sqrt(
                interpolated_values(3) * interpolated_values(3) +
                interpolated_values(4) * interpolated_values(4) +
                interpolated_values(5) * interpolated_values(5) +
                interpolated_values(6) * interpolated_values(6)
            );
            if (norm != 0. && norm != 1.) {
                for (std::size_t i = 3; i < LieGroupComponents; ++i) {
                    interpolated_values(i) /= norm;
                }
            }
        });
    }
    member.team_barrier();
    return interpolated_values;
}

}  // namespace openturbine::gebt_poc