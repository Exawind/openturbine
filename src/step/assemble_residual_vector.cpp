#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "src/beams/beams.hpp"
#include "src/system/integrate_residual_vector.hpp"

namespace openturbine {

void AssembleResidualVector(const Beams& beams) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Residual");
    auto range_policy = Kokkos::TeamPolicy<>(static_cast<int>(beams.num_elems), Kokkos::AUTO());
    const auto shape_size =
        Kokkos::View<double**>::shmem_size(beams.max_elem_nodes, beams.max_elem_qps);
    const auto weight_size = Kokkos::View<double*>::shmem_size(beams.max_elem_qps);
    const auto node_variable_size = Kokkos::View<double* [6]>::shmem_size(beams.max_elem_nodes);
    const auto qp_variable_size = Kokkos::View<double* [6]>::shmem_size(beams.max_elem_qps);
    range_policy.set_scratch_size(
        1,
        Kokkos::PerTeam(2 * shape_size + 2 * weight_size + node_variable_size + 4 * qp_variable_size)
    );
    Kokkos::parallel_for(
        "IntegrateResidualVector", range_policy,
        IntegrateResidualVector{
            beams.num_nodes_per_element, beams.num_qps_per_element, beams.qp_weight,
            beams.qp_jacobian, beams.shape_interp, beams.shape_deriv, beams.node_FX, beams.qp_Fc,
            beams.qp_Fd, beams.qp_Fi, beams.qp_Fg, beams.residual_vector_terms}
    );
}

}  // namespace openturbine
