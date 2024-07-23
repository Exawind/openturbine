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

    Kokkos::parallel_for(
        "IntegrateInertiaMatrix", range_policy,
        IntegrateInertiaMatrix{
            beams.elem_indices, beams.qp_weight, beams.qp_jacobian, beams.shape_interp, beams.qp_Muu,
            beams.qp_Guu, beta_prime, gamma_prime, M}
    );
}

}  // namespace openturbine
