#pragma once

#include <array>
#include <vector>

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

#include "compute_num_system_dofs.hpp"
#include "constraints/constraint_type.hpp"
#include "create_full_matrix.hpp"
#include "linear_solver/dss_handle.hpp"
#include "linear_solver/dss_symbolic.hpp"

#include "OpenTurbine_config.h"

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
template <typename DeviceType>
struct Solver {
    // Define some types for the solver to make the code more readable
    using ValueType = double;
#if defined(OpenTurbine_ENABLE_MKL) && !defined(KOKKOS_ENABLE_CUDA)
    using IndexType = MKL_INT;
#else
    using IndexType = int;
#endif
    using CrsMatrixType = KokkosSparse::CrsMatrix<ValueType, IndexType, DeviceType, void, IndexType>;
    using MultiVectorType = Kokkos::View<ValueType* [1], Kokkos::LayoutLeft, DeviceType>;
#ifdef KOKKOS_ENABLE_CUDA
#if defined(OpenTurbine_ENABLE_CUDSS)
    using HandleType = DSSHandle<DSSAlgorithm::CUDSS>;
#elif defined(OpenTurbine_ENABLE_CUSOLVERSP)
    using HandleType = DSSHandle<DSSAlgorithm::CUSOLVER_SP>;
#elif defined(OpenTurbine_ENABLE_MKL)
    using HandleType = DSSHandle<DSSAlgorithm::MKL>;
#elif defined(OpenTurbine_ENABLE_SUPERLU_MT)
    using HandleType = DSSHandle<DSSAlgorithm::SUPERLU_MT>;
#elif defined(OpenTurbine_ENABLE_KLU)
    using HandleType = DSSHandle<DSSAlgorithm::KLU>;
#elif defined(OpenTurbine_ENABLE_UMFPACK)
    using HandleType = DSSHandle<DSSAlgorithm::UMFPACK>;
#elif defined(OpenTurbine_ENABLE_SUPERLU)
    using HandleType = DSSHandle<DSSAlgorithm::SUPERLU>;
#endif
#elif defined(KOKKOS_ENABLE_HIP)
#if defined(OpenTurbine_ENABLE_MKL)
    using HandleType = DSSHandle<DSSAlgorithm::MKL>;
#elif defined(OpenTurbine_ENABLE_SUPERLU_MT)
    using HandleType = DSSHandle<DSSAlgorithm::SUPERLU_MT>;
#elif defined(OpenTurbine_ENABLE_KLU)
    using HandleType = DSSHandle<DSSAlgorithm::KLU>;
#elif defined(OpenTurbine_ENABLE_UMFPACK)
    using HandleType = DSSHandle<DSSAlgorithm::UMFPACK>;
#elif defined(OpenTurbine_ENABLE_SUPERLU)
    using HandleType = DSSHandle<DSSAlgorithm::SUPERLU>;
#endif
#else
#if defined(OpenTurbine_ENABLE_MKL)
    using HandleType = DSSHandle<DSSAlgorithm::MKL>;
#elif defined(OpenTurbine_ENABLE_SUPERLU_MT)
    using HandleType = DSSHandle<DSSAlgorithm::SUPERLU_MT>;
#elif defined(OpenTurbine_ENABLE_KLU)
    using HandleType = DSSHandle<DSSAlgorithm::KLU>;
#elif defined(OpenTurbine_ENABLE_UMFPACK)
    using HandleType = DSSHandle<DSSAlgorithm::UMFPACK>;
#elif defined(OpenTurbine_ENABLE_SUPERLU)
    using HandleType = DSSHandle<DSSAlgorithm::SUPERLU>;
#endif
#endif

    size_t num_system_nodes;  //< Number of system nodes
    size_t num_system_dofs;   //< Number of system degrees of freedom
    size_t num_dofs;          //< Number of degrees of freedom

    CrsMatrixType A;    //< System matrix
    MultiVectorType b;  //< System RHS
    MultiVectorType x;  //< System solution

    HandleType handle;

    std::vector<double> convergence_err;

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
        const typename Kokkos::View<size_t*, DeviceType>::const_type& node_IDs,
        const typename Kokkos::View<size_t*, DeviceType>::const_type& active_dofs,
        const typename Kokkos::View<size_t*, DeviceType>::const_type& node_freedom_map_table,
        const typename Kokkos::View<size_t*, DeviceType>::const_type& num_nodes_per_element,
        const typename Kokkos::View<size_t**, DeviceType>::const_type& node_state_indices,
        size_t num_constraint_dofs,
        const typename Kokkos::View<size_t*, DeviceType>::const_type& base_active_dofs,
        const typename Kokkos::View<size_t*, DeviceType>::const_type& target_active_dofs,
        const typename Kokkos::View<size_t* [6], DeviceType>::const_type&
            constraint_base_node_freedom_table,
        const typename Kokkos::View<size_t* [6], DeviceType>::const_type&
            constraint_target_node_freedom_table,
        const typename Kokkos::View<Kokkos::pair<size_t, size_t>*, DeviceType>::const_type&
            constraint_row_range
    )
        : num_system_nodes(node_IDs.extent(0)),
          num_system_dofs(ComputeNumSystemDofs<DeviceType>(active_dofs)),
          num_dofs(num_system_dofs + num_constraint_dofs),
          A(CreateFullMatrix<CrsMatrixType>(
              num_system_dofs, num_dofs, base_active_dofs, target_active_dofs,
              constraint_base_node_freedom_table, constraint_target_node_freedom_table,
              constraint_row_range, active_dofs, node_freedom_map_table, num_nodes_per_element,
              node_state_indices
          )),
          b(Kokkos::view_alloc("b", Kokkos::WithoutInitializing), num_dofs),
          x(Kokkos::view_alloc("x", Kokkos::WithoutInitializing), num_dofs) {
        dss_symbolic(handle, A);
    }
};

}  // namespace openturbine
