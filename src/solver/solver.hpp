#pragma once

#include <array>
#include <vector>

#include <Amesos2.hpp>
#include <KokkosSparse.hpp>
#include <KokkosSparse_spadd.hpp>
#include <Kokkos_Core.hpp>

#include "compute_num_system_dofs.hpp"
#include "constraints/constraint_type.hpp"
#include "create_b_matrix.hpp"
#include "create_b_t_matrix.hpp"
#include "create_constraints_matrix_full.hpp"
#include "create_global_matrix.hpp"
#include "create_k_matrix.hpp"
#include "create_matrix_spadd.hpp"
#include "create_matrix_spgemm.hpp"
#include "create_sparse_dense_solver.hpp"
#include "create_system_matrix_full.hpp"
#include "create_t_matrix.hpp"
#include "create_transpose_matrix_full.hpp"
#include "fill_unshifted_row_ptrs.hpp"

namespace openturbine {

/** @brief A linear systems solver for assembling and solving system matrices and constraints in
 * OpenTurbine. This wraps Trilinos packages (Tpetra, Amesos2) for handling sparse matrix
 * operations and linear systems solution.
 *
 * @details This solver manages the assembly and solution of linear systems arising from the
 * generalized-alpha based time integration of the dynamic structural problem. The linear systems
 * include:
 *   - System stiffness matrix (K)
 *   - Tangent operator matrix (T)
 *   - Constraint matrices (B and B_t)
 *   - Combined system matrices for the complete structural problem
 */
struct Solver {
    // Define some types for the solver to make the code more readable
    using GlobalCrsMatrixType = Tpetra::CrsMatrix<>;
    using ExecutionSpace = GlobalCrsMatrixType::execution_space;
    using MemorySpace = GlobalCrsMatrixType::memory_space;
    using GlobalMultiVectorType = Tpetra::MultiVector<>;
    using CrsMatrixType = GlobalCrsMatrixType::local_matrix_device_type;
    using KernelHandle = typename KokkosKernels::Experimental::KokkosKernelsHandle<
        CrsMatrixType::const_size_type, CrsMatrixType::const_ordinal_type,
        CrsMatrixType::const_value_type, ExecutionSpace, MemorySpace, MemorySpace>;

    size_t num_system_nodes;  //< Number of system nodes
    size_t num_system_dofs;   //< Number of system degrees of freedom
    size_t num_dofs;          //< Number of degrees of freedom

    KernelHandle spc_spadd_handle;
    KernelHandle full_system_spadd_handle;

    CrsMatrixType B_t;                         //< Transpose of constraints matrix
    CrsMatrixType system_matrix;               //< System matrix
    CrsMatrixType constraints_matrix;          //< Constraints matrix
    CrsMatrixType system_matrix_full;          //< System matrix with constraints
    CrsMatrixType constraints_matrix_full;     //< Constraints matrix with system
    CrsMatrixType transpose_matrix_full;       //< Transpose of system matrix with constraints
    CrsMatrixType system_plus_constraints;     //< System matrix with constraints
    CrsMatrixType full_matrix;                 //< Full system matrix
    Teuchos::RCP<GlobalCrsMatrixType> A;       //< System matrix
    Teuchos::RCP<GlobalMultiVectorType> b;     //< System RHS
    Teuchos::RCP<GlobalMultiVectorType> x_mv;  //< System solution
    Kokkos::View<double*> R;                   //< System residual vector
    Kokkos::View<double*> x;                   //< System solution vector

    std::vector<double> convergence_err;

    Teuchos::RCP<Amesos2::Solver<GlobalCrsMatrixType, GlobalMultiVectorType>> amesos_solver;

    /** @brief Constructs a sparse linear systems solver for OpenTurbine
     *
     * @param node_IDs View containing the global IDs for each node
     * @param node_freedom_allocation_table View containing the freedom signature for each node
     * @param node_freedom_map_table View mapping node indices to DOF indices
     * @param num_nodes_per_element View containing number of nodes per element
     * @param node_state_indices View containing element-to-node connectivity
     * @param num_constraint_dofs Number of constraint degrees of freedom
     * @param constraint_type View containing the type of each constraint
     * @param constraint_base_node_freedom_table View containing base node DOFs for constraints
     * @param constraint_target_node_freedom_table View containing target node DOFs for constraints
     * @param constraint_row_range View containing row ranges for each constraint
     * @param constraint_base_node_col_range View containing col ranges for base node of each
     * constraint
     * @param constraint_target_node_col_range View containing col ranges for target node of each
     * constraint
     */
    Solver(
        const Kokkos::View<size_t*>::const_type& node_IDs,
        const Kokkos::View<FreedomSignature*>::const_type& node_freedom_allocation_table,
        const Kokkos::View<size_t*>::const_type& node_freedom_map_table,
        const Kokkos::View<size_t*>::const_type& num_nodes_per_element,
        const Kokkos::View<size_t**>::const_type& node_state_indices, size_t num_constraint_dofs,
        const Kokkos::View<ConstraintType*>::const_type& constraint_type,
        const Kokkos::View<size_t* [6]>::const_type& constraint_base_node_freedom_table,
        const Kokkos::View<size_t* [6]>::const_type& constraint_target_node_freedom_table,
        const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& constraint_row_range,
        const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type&
            constraint_base_node_col_range,
        const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type&
            constraint_target_node_col_range
    )
        : num_system_nodes(node_IDs.extent(0)),
          num_system_dofs(ComputeNumSystemDofs(node_freedom_allocation_table)),
          num_dofs(num_system_dofs + num_constraint_dofs),
          B_t(CreateBtMatrix<CrsMatrixType>(
              num_system_dofs, num_constraint_dofs, constraint_type,
              constraint_base_node_freedom_table, constraint_target_node_freedom_table,
              constraint_row_range, constraint_base_node_col_range, constraint_target_node_col_range
          )),
          system_matrix(CreateKMatrix<CrsMatrixType>(
              num_system_dofs, node_freedom_allocation_table, node_freedom_map_table,
              num_nodes_per_element, node_state_indices
          )),
          constraints_matrix(CreateBMatrix<CrsMatrixType>(
              num_system_dofs, num_constraint_dofs, constraint_type,
              constraint_base_node_freedom_table, constraint_target_node_freedom_table,
              constraint_row_range, constraint_base_node_col_range, constraint_target_node_col_range
          )),
          system_matrix_full(CreateSystemMatrixFull(num_system_dofs, num_dofs, system_matrix)),
          constraints_matrix_full(
              CreateConstraintsMatrixFull(num_system_dofs, num_dofs, constraints_matrix)
          ),
          transpose_matrix_full(CreateTransposeMatrixFull(num_system_dofs, num_dofs, B_t)),
          system_plus_constraints(
              CreateMatrixSpadd(system_matrix_full, constraints_matrix_full, spc_spadd_handle)
          ),
          full_matrix(CreateMatrixSpadd(
              system_plus_constraints, transpose_matrix_full, full_system_spadd_handle
          )),
          A(CreateGlobalMatrix<GlobalCrsMatrixType, GlobalMultiVectorType>(full_matrix)),
          b(Tpetra::createMultiVector<GlobalCrsMatrixType::scalar_type>(A->getRangeMap(), 1)),
          x_mv(Tpetra::createMultiVector<GlobalCrsMatrixType::scalar_type>(A->getDomainMap(), 1)),
          R("R", num_dofs),
          x("x", num_dofs),
          amesos_solver(
              CreateSparseDenseSolver<GlobalCrsMatrixType, GlobalMultiVectorType>(A, x_mv, b)
          ) {}

    // cppcheck-suppress missingMemberCopy
    Solver(const Solver& other)
        : num_system_nodes(other.num_system_nodes),
          num_system_dofs(other.num_system_dofs),
          num_dofs(other.num_dofs),
          B_t("B_t", other.B_t),
          system_matrix("system_matrix", other.system_matrix),
          constraints_matrix(other.constraints_matrix),
          system_matrix_full(CreateSystemMatrixFull(num_system_dofs, num_dofs, system_matrix)),
          constraints_matrix_full(
              CreateConstraintsMatrixFull(num_system_dofs, num_dofs, constraints_matrix)
          ),
          transpose_matrix_full(CreateTransposeMatrixFull(num_system_dofs, num_dofs, B_t)),
          system_plus_constraints(
              CreateMatrixSpadd(system_matrix_full, constraints_matrix_full, spc_spadd_handle)
          ),
          full_matrix(CreateMatrixSpadd(
              system_plus_constraints, transpose_matrix_full, full_system_spadd_handle
          )),
          A(CreateGlobalMatrix<GlobalCrsMatrixType, GlobalMultiVectorType>(full_matrix)),
          b(Tpetra::createMultiVector<GlobalCrsMatrixType::scalar_type>(A->getRangeMap(), 1)),
          x_mv(Tpetra::createMultiVector<GlobalCrsMatrixType::scalar_type>(A->getDomainMap(), 1)),
          R("R", num_dofs),
          x("x", num_dofs),
          convergence_err(other.convergence_err),
          amesos_solver(
              CreateSparseDenseSolver<GlobalCrsMatrixType, GlobalMultiVectorType>(A, x_mv, b)
          ) {}

    Solver(Solver&& other) noexcept = delete;
    ~Solver() = default;

    Solver& operator=(const Solver& other) {
        if (this == &other) {
            return *this;
        }

        auto tmp = other;
        std::swap(num_system_nodes, tmp.num_system_nodes);
        std::swap(num_system_dofs, tmp.num_system_dofs);
        std::swap(num_dofs, tmp.num_dofs);

        std::swap(spc_spadd_handle, tmp.spc_spadd_handle);
        std::swap(full_system_spadd_handle, tmp.full_system_spadd_handle);

        std::swap(B_t, tmp.B_t);
        std::swap(system_matrix, tmp.system_matrix);
        std::swap(constraints_matrix, tmp.constraints_matrix);
        std::swap(system_matrix_full, tmp.system_matrix_full);
        std::swap(constraints_matrix_full, tmp.constraints_matrix_full);
        std::swap(transpose_matrix_full, tmp.transpose_matrix_full);
        std::swap(system_plus_constraints, tmp.system_plus_constraints);
        std::swap(full_matrix, tmp.full_matrix);
        std::swap(A, tmp.A);
        std::swap(b, tmp.b);
        std::swap(x_mv, tmp.x_mv);
        std::swap(R, tmp.R);
        std::swap(x, tmp.x);

        std::swap(convergence_err, tmp.convergence_err);

        std::swap(amesos_solver, tmp.amesos_solver);
        return *this;
    }

    Solver& operator=(Solver&& other) noexcept = delete;
};

}  // namespace openturbine
