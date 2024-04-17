#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "AssembleElasticStiffnessMatrix.hpp"
#include "AssembleGyroscopicInertiaMatrix.hpp"
#include "AssembleInertialStiffnessMatrix.hpp"
#include "AssembleMassMatrix.hpp"
#include "AssembleResidualVector.hpp"
#include "CalculateTangentOperator.hpp"
#include "Solver.hpp"
#include "beams.hpp"

namespace openturbine {

template <typename Subview_NxN, typename Subview_N>
void AssembleSystem(Solver& solver, Beams& beams, Subview_NxN St_11, Subview_N R_system) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble System");
    Kokkos::deep_copy(solver.T, 0.);
    Kokkos::parallel_for(
        "TangentOperator", solver.num_system_nodes,
        CalculateTangentOperator{
            solver.h,
            solver.state.q_delta,
            solver.T,
        }
    );

    Kokkos::deep_copy(R_system, 0.);
    AssembleResidualVector(beams, R_system);

    Kokkos::deep_copy(solver.K, 0.);
    AssembleElasticStiffnessMatrix(beams, solver.K);
    AssembleInertialStiffnessMatrix(beams, solver.K);

    KokkosBlas::gemm("N", "N", 1.0, solver.K, solver.T, 0.0, St_11);

    if (solver.is_dynamic_solve) {
        Kokkos::deep_copy(solver.M, 0.);
        Kokkos::deep_copy(solver.G, 0.);
        AssembleMassMatrix(beams, solver.M);
        AssembleGyroscopicInertiaMatrix(beams, solver.G);
        KokkosBlas::axpy(solver.beta_prime, solver.M, St_11);
        KokkosBlas::axpy(solver.gamma_prime, solver.G, St_11);
    }
}

}  // namespace openturbine
