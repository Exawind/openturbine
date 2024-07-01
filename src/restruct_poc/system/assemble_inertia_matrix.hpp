#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "integrate_inertia_matrix.hpp"

#include "src/restruct_poc/beams/beams.hpp"

namespace openturbine {

inline void AssembleInertiaMatrix(
    Beams& beams, double beta_prime, double gamma_prime, Kokkos::View<double***> M
) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Inertia Matrix");
    auto range_policy = Kokkos::TeamPolicy<>(beams.num_elems, Kokkos::AUTO());
    // const auto max_elem_qp = beams.max_elem_qps;
    // const auto max_elem_nodes = beams.max_elem_nodes;
    // const auto coeff_size =  Kokkos::View<double***>::shmem_size(max_elem_nodes, max_elem_nodes,
    // max_elem_qp); range_policy.set_scratch_size(1, Kokkos::PerTeam(coeff_size));

    Kokkos::parallel_for(
        "IntegrateInertiaMatrix", range_policy,
        IntegrateInertiaMatrix{
            beams.elem_indices, beams.qp_weight, beams.qp_jacobian, beams.shape_interp, beams.qp_Muu,
            beams.qp_Guu, beta_prime, gamma_prime, M}
    );

    // Kokkos::parallel_for(
    //     "IntegrateInertiaMatrix", Kokkos::MDRangePolicy{
    //             {0, 0, 0}, {beams.num_elems, beams.max_elem_nodes, beams.max_elem_nodes}},
    //     IntegrateInertiaMatrix{
    //         beams.elem_indices, beams.qp_weight, beams.qp_jacobian,
    //         beams.shape_interp, beams.qp_Muu, beams.qp_Guu, beta_prime, gamma_prime, M}
    // );
}

}  // namespace openturbine
