#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "src/beams/beams.hpp"
#include "src/system/integrate_inertia_matrix.hpp"

namespace openturbine {

void AssembleInertiaMatrix(const Beams& beams, double beta_prime, double gamma_prime) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Inertia Matrix");
    auto range_policy = Kokkos::TeamPolicy<>(static_cast<int>(beams.num_elems), Kokkos::AUTO());
    auto smem = 2 * Kokkos::View<double* [6][6]>::shmem_size(beams.max_elem_qps) +
                2 * Kokkos::View<double*>::shmem_size(beams.max_elem_qps) +
                Kokkos::View<double**>::shmem_size(beams.max_elem_nodes, beams.max_elem_qps);
    range_policy.set_scratch_size(1, Kokkos::PerTeam(smem));
    Kokkos::parallel_for(
        "IntegrateInertiaMatrix", range_policy,
        IntegrateInertiaMatrix{
            beams.num_nodes_per_element, beams.num_qps_per_element, beams.qp_weight,
            beams.qp_jacobian, beams.shape_interp, beams.qp_Muu, beams.qp_Guu, beta_prime,
            gamma_prime, beams.inertia_matrix_terms}
    );
}

}  // namespace openturbine
