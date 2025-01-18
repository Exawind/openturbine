#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "elements/beams/beams.hpp"
#include "system/beams/integrate_stiffness_matrix.hpp"

namespace openturbine {

inline void AssembleStiffnessMatrixBeams(const Beams& beams) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Stiffness Matrix");
    auto range_policy = Kokkos::TeamPolicy<>(static_cast<int>(beams.num_elems), Kokkos::AUTO());
    auto smem = 5 * Kokkos::View<double* [6][6]>::shmem_size(beams.max_elem_qps) +
                2 * Kokkos::View<double*>::shmem_size(beams.max_elem_qps) +
                2 * Kokkos::View<double**>::shmem_size(beams.max_elem_nodes, beams.max_elem_qps);
    range_policy.set_scratch_size(1, Kokkos::PerTeam(smem));
    Kokkos::parallel_for(
        "IntegrateStiffnessMatrix", range_policy,
        IntegrateStiffnessMatrix{
            beams.num_nodes_per_element,
            beams.num_qps_per_element,
            beams.qp_weight,
            beams.qp_jacobian,
            beams.shape_interp,
            beams.shape_deriv,
            beams.qp_Kuu,
            beams.qp_Puu,
            beams.qp_Cuu,
            beams.qp_Ouu,
            beams.qp_Quu,
            beams.stiffness_matrix_terms,
        }
    );
}

}  // namespace openturbine
