#pragma once

#include <KokkosBlas.hpp>
#include <KokkosSparse.hpp>
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

    auto num_rows = solver.K.extent(0);
    auto num_columns = solver.K.extent(1);
    auto num_non_zero = (beams.max_elem_nodes * 6) * (beams.max_elem_nodes * 6) * beams.num_elems; //make this a reduction
    auto row_ptrs = Kokkos::View<int*>("row_ptrs", num_rows+1);
    auto indices = Kokkos::View<int*>("indices", num_non_zero);
    auto K_values = Kokkos::View<double*>("K values", num_non_zero);
    auto T_values = Kokkos::View<double*>("T values", num_non_zero);
    Kokkos::parallel_for("Populate Sparse Row Ptrs", 1, KOKKOS_LAMBDA(int) {
      auto rows_so_far = 0;
      for(int i_elem = 0; i_elem < beams.num_elems; ++i_elem) {
        auto elem_indices = beams.elem_indices;
        auto idx = elem_indices[i_elem];
        auto num_nodes = idx.num_nodes;
        for(int i = 0; i < num_nodes*kLieAlgebraComponents; ++i) {
            row_ptrs(rows_so_far + 1) = row_ptrs(rows_so_far) + num_nodes * kLieAlgebraComponents;
            ++rows_so_far;
        }
        
      }
    });
    Kokkos::parallel_for("Populate Sparse Indices", 1, KOKKOS_LAMBDA(int) {
      auto entries_so_far = 0;
      for(int i_elem = 0; i_elem < beams.num_elems; ++i_elem) {
        auto node_state_indices = beams.node_state_indices;
        auto elem_indices = beams.elem_indices;
        auto idx = elem_indices[i_elem];
        auto num_nodes = idx.num_nodes;
        for(int j_index = 0; j_index < num_nodes; ++j_index) {
          for(int n = 0; n < kLieAlgebraComponents; ++n) {
            for(int i_index = 0; i_index < num_nodes; ++i_index) {
              const auto i = i_index + idx.node_range.first;
              const auto column_start = node_state_indices(i)*kLieAlgebraComponents;
              for(int m = 0; m < kLieAlgebraComponents; ++m) {
                  indices(entries_so_far) = column_start + m;
                  ++entries_so_far;
              }
            }
          }
        }
      }
    });

    using crs_matrix_type = KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, void, int>;
    auto K = crs_matrix_type("K", num_rows, num_columns, num_non_zero, K_values, row_ptrs, indices);
    auto T = crs_matrix_type("T", num_rows, num_columns, num_non_zero, T_values, row_ptrs, indices);

    auto sk = solver.K;

    auto row_data_size = Kokkos::View<double*>::shmem_size(num_columns);
    auto col_idx_size = Kokkos::View<int*>::shmem_size(num_columns);
    auto sparse_matrix_policy = Kokkos::TeamPolicy<>(num_rows, Kokkos::AUTO());
    sparse_matrix_policy.set_scratch_size(1, Kokkos::PerTeam(row_data_size + col_idx_size));
    Kokkos::parallel_for("Copy into Sparse Matrix", sparse_matrix_policy, KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type member) {
        auto i = member.league_rank();
        auto row = K.row(i);
        auto row_map = K.graph.row_map;
        auto cols = K.graph.entries;
        auto row_data = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(member.team_scratch(1), num_columns);
        auto col_idx = Kokkos::View<int*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(member.team_scratch(1), num_columns);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, row.length), [=](int entry) {
            col_idx(entry) = cols(row_map(i) + entry);
            row_data(entry) = sk(i, col_idx(entry));
        });
        member.team_barrier();
        Kokkos::single(Kokkos::PerTeam(member), [=](){
            K.replaceValues(i, col_idx.data(), row.length, row_data.data());
        });
    });
    
    auto st = solver.T;
    Kokkos::parallel_for("Copy into Sparse Matrix", sparse_matrix_policy, KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type member) {
        auto i = member.league_rank();
        auto row = T.row(i);
        auto row_map = T.graph.row_map;
        auto cols = T.graph.entries;
        auto row_data = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(member.team_scratch(1), num_columns);
        auto col_idx = Kokkos::View<int*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(member.team_scratch(1), num_columns);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, row.length), [=](int entry) {
            col_idx(entry) = cols(row_map(i) + entry);
            row_data(entry) = st(i, col_idx(entry));
        });
        member.team_barrier();
        Kokkos::single(Kokkos::PerTeam(member), [=](){
            T.replaceValues(i, col_idx.data(), row.length, row_data.data());
        });
    });

    Kokkos::deep_copy(St_11, 0.);
    
    crs_matrix_type local_St_11 = KokkosSparse::spgemm<crs_matrix_type>(K, false, T, false);
    Kokkos::parallel_for("Copy into St_11", sparse_matrix_policy, KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type member) {
        auto i = member.league_rank();
        auto row = local_St_11.row(i);
        auto row_map = local_St_11.graph.row_map;
        auto cols = local_St_11.graph.entries;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, row.length), [=](int entry) {
            St_11(i, cols(row_map(i) + entry)) = row.value(entry);
        });
    });   

    if (solver.is_dynamic_solve) {
        Kokkos::deep_copy(solver.K, 0.);
        AssembleMassMatrix(beams, solver.beta_prime, solver.K);
        AssembleGyroscopicInertiaMatrix(beams, solver.gamma_prime, solver.K);
        KokkosBlas::update(0., solver.K, 1., solver.K, 1., St_11);
    }
}

}  // namespace openturbine
