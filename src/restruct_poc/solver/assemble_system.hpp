#pragma once

#include <KokkosBlas.hpp>
#include <KokkosSparse.hpp>
#include <KokkosSparse_spadd.hpp>
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

struct CopyIntoSparseMatrix {
    using crs_matrix_type = KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>, void, int>;
    using row_data_type = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using col_idx_type = Kokkos::View<int*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    crs_matrix_type sparse;
    Kokkos::View<const double**> dense;

    KOKKOS_FUNCTION
    void operator()(Kokkos::TeamPolicy<>::member_type member) const {
        auto i = member.league_rank();
        auto row = sparse.row(i);
        auto row_map = sparse.graph.row_map;
        auto cols = sparse.graph.entries;
        auto row_data = row_data_type(member.team_scratch(1), row.length);
        auto col_idx = col_idx_type(member.team_scratch(1), row.length);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, row.length), [=](int entry) {
            col_idx(entry) = cols(row_map(i) + entry);
            row_data(entry) = dense(i, col_idx(entry));
        });
        member.team_barrier();
        Kokkos::single(Kokkos::PerTeam(member), [=](){
            sparse.replaceValues(i, col_idx.data(), row.length, row_data.data());
        });
    }
};

struct ComputeNumberOfNonZeros {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    
    KOKKOS_FUNCTION
    void operator()(int i_elem, int& update) const {
        auto idx = elem_indices[i_elem];
        auto num_nodes = idx.num_nodes;
        update += (num_nodes*6) * (num_nodes*6);
    }
};

struct PopulateSparseRowPtrs {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    Kokkos::View<int*> row_ptrs;

    KOKKOS_FUNCTION
    void operator()(int) const {
      const auto num_elems = elem_indices.extent(0);
      auto rows_so_far = 0;
      for(int i_elem = 0; i_elem < num_elems; ++i_elem) {
        auto idx = elem_indices[i_elem];
        auto num_nodes = idx.num_nodes;
        for(int i = 0; i < num_nodes*kLieAlgebraComponents; ++i) {
            row_ptrs(rows_so_far + 1) = row_ptrs(rows_so_far) + num_nodes * kLieAlgebraComponents;
            ++rows_so_far;
        }
        
      }
    }
};

struct PopulateSparseIndices {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;
    Kokkos::View<int*>::const_type node_state_indices;    
    Kokkos::View<int*> indices;

    KOKKOS_FUNCTION
    void operator()(int) const {
      const auto num_elems = elem_indices.extent(0);
      auto entries_so_far = 0;
      for(int i_elem = 0; i_elem < num_elems; ++i_elem) {
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
    }

};

template <typename Subview_NxN, typename Subview_N>
void AssembleSystem(Solver& solver, Beams& beams, Subview_NxN St_11, Subview_N R_system) {
    auto region = Kokkos::Profiling::ScopedRegion("Assemble System");
    using KernelHandle = typename KokkosKernels::Experimental::KokkosKernelsHandle<int, int, double, Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space, Kokkos::DefaultExecutionSpace::memory_space>;
    KernelHandle kh;
    kh.create_spgemm_handle();
    kh.create_spadd_handle();

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
    auto num_non_zero = 0;
    Kokkos::parallel_reduce("ComputeNumberOfNonZeros", beams.num_elems, ComputeNumberOfNonZeros{beams.elem_indices}, num_non_zero);

    auto row_ptrs = Kokkos::View<int*>("row_ptrs", num_rows+1);
    auto indices = Kokkos::View<int*>("indices", num_non_zero);
    Kokkos::parallel_for("PopulateSparseRowPtrs", 1, PopulateSparseRowPtrs{beams.elem_indices, row_ptrs});
    Kokkos::parallel_for("Populate Sparse Indices", 1, PopulateSparseIndices{beams.elem_indices, beams.node_state_indices, indices});

    using crs_matrix_type = KokkosSparse::CrsMatrix<double, int, Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>, void, int>;
    Kokkos::fence();
    auto K_values = Kokkos::View<double*>("K values", num_non_zero);
    auto K = crs_matrix_type("K", num_rows, num_columns, num_non_zero, K_values, row_ptrs, indices);
    auto T_values = Kokkos::View<double*>("T values", num_non_zero);
    auto T = crs_matrix_type("T", num_rows, num_columns, num_non_zero, T_values, row_ptrs, indices);

    auto row_data_size = Kokkos::View<double*>::shmem_size(num_columns);
    auto col_idx_size = Kokkos::View<int*>::shmem_size(num_columns);
    auto sparse_matrix_policy = Kokkos::TeamPolicy<>(num_rows, Kokkos::AUTO());
    sparse_matrix_policy.set_scratch_size(1, Kokkos::PerTeam(row_data_size + col_idx_size));

    Kokkos::parallel_for("Copy into Sparse Matrix", sparse_matrix_policy, CopyIntoSparseMatrix{K, solver.K});
    Kokkos::parallel_for("Copy into Sparse Matrix", sparse_matrix_policy, CopyIntoSparseMatrix{T, solver.T});

    Kokkos::fence();
    crs_matrix_type static_system_matrix;
    KokkosSparse::spgemm_symbolic(kh, K, false, T, false, static_system_matrix);
    KokkosSparse::spgemm_numeric(kh, K, false, T, false, static_system_matrix);

    auto beta_prime = (solver.is_dynamic_solve) ? solver.beta_prime : 0.;
    auto gamma_prime = (solver.is_dynamic_solve) ? solver.gamma_prime : 0.;
    Kokkos::deep_copy(solver.K, 0.);
    AssembleMassMatrix(beams, beta_prime, solver.K);
    AssembleGyroscopicInertiaMatrix(beams, gamma_prime, solver.K);
    Kokkos::parallel_for("Copy into Sparse Matrix", sparse_matrix_policy, CopyIntoSparseMatrix{K, solver.K});

    Kokkos::fence();
    crs_matrix_type system_matrix;
    KokkosSparse::spadd_symbolic(&kh, K, static_system_matrix, system_matrix);
    KokkosSparse::spadd_numeric(&kh, 1., K, 1., static_system_matrix, system_matrix);

    Kokkos::deep_copy(St_11, 0.);
    Kokkos::fence();
    Kokkos::parallel_for("Copy into St_11", sparse_matrix_policy, KOKKOS_LAMBDA(Kokkos::TeamPolicy<>::member_type member) {
        auto i = member.league_rank();
        auto row = system_matrix.row(i);
        auto row_map = system_matrix.graph.row_map;
        auto cols = system_matrix.graph.entries;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, row.length), [=](int entry) {
            St_11(i, cols(row_map(i) + entry)) = row.value(entry);
        });
    });   
}

}  // namespace openturbine
