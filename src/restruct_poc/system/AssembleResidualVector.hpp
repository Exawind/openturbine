#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "IntegrateResidualVector.hpp"

#include "src/restruct_poc/beams/Beams.hpp"
#include "src/restruct_poc/types.hpp"

namespace openturbine {

inline void AssembleResidualVector(Beams& beams, View_N residual_vector) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble Residual");
    Kokkos::parallel_for(
        "IntegrateResidualVector", beams.num_nodes,
        IntegrateResidualVector{
            beams.node_state_indices, beams.node_FE, beams.node_FI, beams.node_FG, beams.node_FX,
            residual_vector}
    );
}

}  // namespace openturbine
