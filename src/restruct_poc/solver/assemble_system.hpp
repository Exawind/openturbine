#pragma once

#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "solver.hpp"

#include "src/restruct_poc/beams/beams.hpp"
#include "src/restruct_poc/system/assemble_elastic_stiffness_matrix.hpp"
#include "src/restruct_poc/system/assemble_gyroscopic_inertia_matrix.hpp"
#include "src/restruct_poc/system/assemble_inertial_stiffness_matrix.hpp"
#include "src/restruct_poc/system/assemble_mass_matrix.hpp"
#include "src/restruct_poc/system/assemble_residual_vector.hpp"
#include "src/restruct_poc/system/calculate_tangent_operator.hpp"

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
        Kokkos::deep_copy(solver.K, 0.);
        AssembleMassMatrix(beams, solver.beta_prime, solver.K);
        AssembleGyroscopicInertiaMatrix(beams, solver.gamma_prime, solver.K);
        KokkosBlas::update(0., solver.K, 1., solver.K, 1., St_11);
    }
}

}  // namespace openturbine
