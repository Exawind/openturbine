#pragma once

#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"
#include "types.hpp"

namespace openturbine {

/**
 * @brief Contains the field variables needed to compute the per-element contributions to the
 * residual vector and system matrix.
 *
 * - In all Views, the first dimension corresponds to the element number
 * - In Views prefixed `node_`, the second dimension corresponds to the node number within that
 * element
 * - In Views prefixed `qp_`, the second dimension corresponds to the quadrature point within that
 * element
 * - The remaining dimensions are the number of components as defined by physics
 * - Additionally, shape_interp and shape_deriv have dimensions num_elems x num_nodes x num_qps
 */
template <typename Device>
struct Beams {
    size_t num_elems;       // Total number of element
    size_t max_elem_nodes;  // Maximum number of nodes per element
    size_t max_elem_qps;    // Maximum number of quadrature points per element

    Kokkos::View<size_t*, Device> num_nodes_per_element;
    Kokkos::View<size_t*, Device> num_qps_per_element;
    Kokkos::View<size_t**, Device> node_state_indices;  // State row index for each node
    Kokkos::View<FreedomSignature**, Device> element_freedom_signature;
    Kokkos::View<size_t** [6], Device> element_freedom_table;

    Kokkos::View<double[3], Device> gravity;

    // Node-based data
    Kokkos::View<double** [7], Device> node_x0;      // Inital position/rotation
    Kokkos::View<double** [7], Device> node_u;       // State: translation/rotation displacement
    Kokkos::View<double** [6], Device> node_u_dot;   // State: translation/rotation velocity
    Kokkos::View<double** [6], Device> node_u_ddot;  // State: translation/rotation acceleration
    Kokkos::View<double** [6], Device> node_FX;      // External forces

    // Quadrature point data
    Kokkos::View<double**, Device> qp_weight;           // Integration weights
    Kokkos::View<double**, Device> qp_jacobian;         // Jacobian vector
    Kokkos::View<double** [6][6], Device> qp_Mstar;     // Mass matrix in material frame
    Kokkos::View<double** [6][6], Device> qp_Cstar;     // Stiffness matrix in material frame
    Kokkos::View<double** [7], Device> qp_x;            // Current position/orientation
    Kokkos::View<double** [3], Device> qp_x0;           // Initial position
    Kokkos::View<double** [3], Device> qp_x0_prime;     // Initial position derivative
    Kokkos::View<double** [4], Device> qp_r0;           // Initial rotation
    Kokkos::View<double** [3], Device> qp_u;            // State: translation displacement
    Kokkos::View<double** [3], Device> qp_u_prime;      // State: translation displacement derivative
    Kokkos::View<double** [3], Device> qp_u_dot;        // State: translation velocity
    Kokkos::View<double** [3], Device> qp_u_ddot;       // State: translation acceleration
    Kokkos::View<double** [4], Device> qp_r;            // State: rotation
    Kokkos::View<double** [4], Device> qp_r_prime;      // State: rotation derivative
    Kokkos::View<double** [3], Device> qp_omega;        // State: angular velocity
    Kokkos::View<double** [3], Device> qp_omega_dot;    // State: position/rotation
    Kokkos::View<double** [3], Device> qp_deformation;  // Deformation relative to rigid body motion
    Kokkos::View<double** [3][4], Device> qp_E;         // Quaternion derivative
    Kokkos::View<double** [6], Device> qp_Fe;           // External force

    Kokkos::View<double** [6], Device> residual_vector_terms;
    Kokkos::View<double*** [6][6], Device> system_matrix_terms;

    // Shape Function data
    Kokkos::View<double***, Device> shape_interp;  // Shape function values
    Kokkos::View<double***, Device> shape_deriv;   // Shape function derivatives

    // Constructor which initializes views based on given sizes
    Beams(const size_t n_beams, const size_t max_e_nodes, const size_t max_e_qps)
        : num_elems(n_beams),
          max_elem_nodes(max_e_nodes),
          max_elem_qps(max_e_qps),
          // Element Data
          num_nodes_per_element(
              Kokkos::view_alloc("num_nodes_per_element", Kokkos::WithoutInitializing), num_elems
          ),
          num_qps_per_element(
              Kokkos::view_alloc("num_qps_per_element", Kokkos::WithoutInitializing), num_elems
          ),
          node_state_indices(
              Kokkos::view_alloc("node_state_indices", Kokkos::WithoutInitializing), num_elems,
              max_elem_nodes
          ),
          element_freedom_signature(
              Kokkos::view_alloc("element_freedom_signature", Kokkos::WithoutInitializing),
              num_elems, max_elem_nodes
          ),
          element_freedom_table(
              Kokkos::view_alloc("element_freedom_table", Kokkos::WithoutInitializing), num_elems,
              max_elem_nodes
          ),
          gravity(Kokkos::view_alloc("gravity", Kokkos::WithoutInitializing)),
          // Node Data
          node_x0(
              Kokkos::view_alloc("node_x0", Kokkos::WithoutInitializing), num_elems, max_elem_nodes
          ),
          node_u(
              Kokkos::view_alloc("node_u", Kokkos::WithoutInitializing), num_elems, max_elem_nodes
          ),
          node_u_dot(
              Kokkos::view_alloc("node_u_dot", Kokkos::WithoutInitializing), num_elems,
              max_elem_nodes
          ),
          node_u_ddot(
              Kokkos::view_alloc("node_u_ddot", Kokkos::WithoutInitializing), num_elems,
              max_elem_nodes
          ),
          node_FX(
              Kokkos::view_alloc("node_force_external", Kokkos::WithoutInitializing), num_elems,
              max_elem_nodes
          ),
          // Quadrature Point data
          qp_weight(
              Kokkos::view_alloc("qp_weight", Kokkos::WithoutInitializing), num_elems, max_elem_qps
          ),
          qp_jacobian(
              Kokkos::view_alloc("qp_jacobian", Kokkos::WithoutInitializing), num_elems, max_elem_qps
          ),
          qp_Mstar(
              Kokkos::view_alloc("qp_Mstar", Kokkos::WithoutInitializing), num_elems, max_elem_qps
          ),
          qp_Cstar(
              Kokkos::view_alloc("qp_Cstar", Kokkos::WithoutInitializing), num_elems, max_elem_qps
          ),
          qp_x(Kokkos::view_alloc("qp_x", Kokkos::WithoutInitializing), num_elems, max_elem_qps),
          qp_x0(Kokkos::view_alloc("qp_x0", Kokkos::WithoutInitializing), num_elems, max_elem_qps),
          qp_x0_prime(
              Kokkos::view_alloc("qp_x0_prime", Kokkos::WithoutInitializing), num_elems, max_elem_qps
          ),
          qp_r0(Kokkos::view_alloc("qp_r0", Kokkos::WithoutInitializing), num_elems, max_elem_qps),
          qp_u(Kokkos::view_alloc("qp_u", Kokkos::WithoutInitializing), num_elems, max_elem_qps),
          qp_u_prime(
              Kokkos::view_alloc("qp_u_prime", Kokkos::WithoutInitializing), num_elems, max_elem_qps
          ),
          qp_u_dot(
              Kokkos::view_alloc("qp_u_dot", Kokkos::WithoutInitializing), num_elems, max_elem_qps
          ),
          qp_u_ddot(
              Kokkos::view_alloc("qp_u_ddot", Kokkos::WithoutInitializing), num_elems, max_elem_qps
          ),
          qp_r(Kokkos::view_alloc("qp_r", Kokkos::WithoutInitializing), num_elems, max_elem_qps),
          qp_r_prime(
              Kokkos::view_alloc("qp_r_prime", Kokkos::WithoutInitializing), num_elems, max_elem_qps
          ),
          qp_omega(
              Kokkos::view_alloc("qp_omega", Kokkos::WithoutInitializing), num_elems, max_elem_qps
          ),
          qp_omega_dot(
              Kokkos::view_alloc("qp_omega_dot", Kokkos::WithoutInitializing), num_elems,
              max_elem_qps
          ),
          qp_deformation(
              Kokkos::view_alloc("qp_deformation", Kokkos::WithoutInitializing), num_elems,
              max_elem_qps
          ),
          qp_Fe(Kokkos::view_alloc("qp_Fe", Kokkos::WithoutInitializing), num_elems, max_elem_qps),
          residual_vector_terms(
              Kokkos::view_alloc("residual_vector_terms", Kokkos::WithoutInitializing), num_elems,
              max_elem_nodes
          ),
          system_matrix_terms(
              Kokkos::view_alloc("system_matrix_terms", Kokkos::WithoutInitializing), num_elems,
              max_elem_nodes, max_elem_nodes
          ),
          // Shape Function data
          shape_interp(
              Kokkos::view_alloc("shape_interp", Kokkos::WithoutInitializing), num_elems,
              max_elem_nodes, max_elem_qps
          ),
          shape_deriv(
              Kokkos::view_alloc("deriv_interp", Kokkos::WithoutInitializing), num_elems,
              max_elem_nodes, max_elem_qps
          ) {
        Kokkos::deep_copy(element_freedom_signature, FreedomSignature::AllComponents);
    }
};

}  // namespace openturbine
