#pragma once

#include <array>
#include <vector>

#include <Amesos2.hpp>
#include <KokkosSparse.hpp>
#include <KokkosSparse_spadd.hpp>
#include <Kokkos_Core.hpp>

#include "compute_num_system_dofs.hpp"
#include "constraints/constraint_type.hpp"
#include "create_full_matrix.hpp"
#include "create_sparse_dense_solver.hpp"

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

    Teuchos::RCP<GlobalCrsMatrixType> A;    //< System matrix
    Teuchos::RCP<GlobalMultiVectorType> b;  //< System RHS
    Teuchos::RCP<GlobalMultiVectorType> x;  //< System solution

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
     * @param base_node_freedom_signature View containing the base node freedom signature of the
     * constraints
     * @param target_node_freedom_signature View containing the target node freedom signature of the
     * constraints
     * @param constraint_base_node_freedom_table View containing base node DOFs for constraints
     * @param constraint_target_node_freedom_table View containing target node DOFs for constraints
     * @param constraint_row_range View containing row ranges for each constraint
     */
    Solver(
        const Kokkos::View<size_t*>::const_type& node_IDs,
        const Kokkos::View<size_t*>::const_type& active_dofs,
        const Kokkos::View<size_t*>::const_type& node_freedom_map_table,
        const Kokkos::View<size_t*>::const_type& num_nodes_per_element,
        const Kokkos::View<size_t**>::const_type& node_state_indices, size_t num_constraint_dofs,
        const Kokkos::View<ConstraintType*>::const_type& constraint_type,
        const Kokkos::View<FreedomSignature*>::const_type& base_node_freedom_signature,
        const Kokkos::View<FreedomSignature*>::const_type& target_node_freedom_signature,
        const Kokkos::View<size_t* [6]>::const_type& constraint_base_node_freedom_table,
        const Kokkos::View<size_t* [6]>::const_type& constraint_target_node_freedom_table,
        const Kokkos::View<Kokkos::pair<size_t, size_t>*>::const_type& constraint_row_range
    )
        : num_system_nodes(node_IDs.extent(0)),
          num_system_dofs(ComputeNumSystemDofs(active_dofs)),
          num_dofs(num_system_dofs + num_constraint_dofs),
          A(CreateFullMatrix<GlobalCrsMatrixType>(
              num_system_dofs, num_dofs, num_constraint_dofs, constraint_type,
              base_node_freedom_signature, target_node_freedom_signature,
              constraint_base_node_freedom_table, constraint_target_node_freedom_table,
              constraint_row_range, active_dofs, node_freedom_map_table,
              num_nodes_per_element, node_state_indices
          )),
          b(Tpetra::createMultiVector<GlobalCrsMatrixType::scalar_type>(A->getRangeMap(), 1)),
          x(Tpetra::createMultiVector<GlobalCrsMatrixType::scalar_type>(A->getDomainMap(), 1)),
          amesos_solver(CreateSparseDenseSolver<GlobalCrsMatrixType, GlobalMultiVectorType>(A, x, b)
          ) {}

    // cppcheck-suppress missingMemberCopy
    Solver(const Solver& other)
        : num_system_nodes(other.num_system_nodes),
          num_system_dofs(other.num_system_dofs),
          num_dofs(other.num_dofs),
          A(other.A),
          b(other.b),
          x(other.x),
          convergence_err(other.convergence_err),
          amesos_solver(CreateSparseDenseSolver<GlobalCrsMatrixType, GlobalMultiVectorType>(A, x, b)
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
        std::swap(A, tmp.A);
        std::swap(b, tmp.b);
        std::swap(x, tmp.x);
        std::swap(convergence_err, tmp.convergence_err);
        std::swap(amesos_solver, tmp.amesos_solver);
        return *this;
    }

    Solver& operator=(Solver&& other) noexcept = delete;
};

}  // namespace openturbine
