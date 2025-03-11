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
struct Beams {
    size_t num_elems;       // Total number of element
    size_t max_elem_nodes;  // Maximum number of nodes per element
    size_t max_elem_qps;    // Maximum number of quadrature points per element

    Kokkos::View<size_t*> num_nodes_per_element;
    Kokkos::View<size_t*> num_qps_per_element;
    Kokkos::View<size_t**> node_state_indices;  // State row index for each node
    Kokkos::View<FreedomSignature**> element_freedom_signature;
    Kokkos::View<size_t** [6]> element_freedom_table;

    View_3 gravity;

    // Node-based data
    Kokkos::View<double** [7]> node_x0;      // Inital position/rotation
    Kokkos::View<double** [7]> node_u;       // State: translation/rotation displacement
    Kokkos::View<double** [6]> node_u_dot;   // State: translation/rotation velocity
    Kokkos::View<double** [6]> node_u_ddot;  // State: translation/rotation acceleration
    Kokkos::View<double** [6]> node_FX;      // External forces

    // Quadrature point data
    Kokkos::View<double**> qp_weight;           // Integration weights
    Kokkos::View<double**> qp_jacobian;         // Jacobian vector
    Kokkos::View<double** [6][6]> qp_Mstar;     // Mass matrix in material frame
    Kokkos::View<double** [6][6]> qp_Cstar;     // Stiffness matrix in material frame
    Kokkos::View<double** [7]> qp_x;            // Current position/orientation
    Kokkos::View<double** [3]> qp_x0;           // Initial position
    Kokkos::View<double** [3]> qp_x0_prime;     // Initial position derivative
    Kokkos::View<double** [4]> qp_r0;           // Initial rotation
    Kokkos::View<double** [3]> qp_u;            // State: translation displacement
    Kokkos::View<double** [3]> qp_u_prime;      // State: translation displacement derivative
    Kokkos::View<double** [3]> qp_u_dot;        // State: translation velocity
    Kokkos::View<double** [3]> qp_u_ddot;       // State: translation acceleration
    Kokkos::View<double** [4]> qp_r;            // State: rotation
    Kokkos::View<double** [4]> qp_r_prime;      // State: rotation derivative
    Kokkos::View<double** [3]> qp_omega;        // State: angular velocity
    Kokkos::View<double** [3]> qp_omega_dot;    // State: position/rotation
    Kokkos::View<double** [3]> qp_deformation;  // Deformation relative to rigid body motion
    Kokkos::View<double** [3][4]> qp_E;         // Quaternion derivative
    Kokkos::View<double** [6]> qp_Fe;           // External force

    Kokkos::View<double** [6]> residual_vector_terms;
    Kokkos::View<double*** [6][6]> system_matrix_terms;

    // Shape Function data
    Kokkos::View<double***> shape_interp;  // Shape function values
    Kokkos::View<double***> shape_deriv;   // Shape function derivatives

    // Constructor which initializes views based on given sizes
    Beams(const size_t n_beams, const size_t max_e_nodes, const size_t max_e_qps)
        : num_elems(n_beams),
          max_elem_nodes(max_e_nodes),
          max_elem_qps(max_e_qps),
          // Element Data
          num_nodes_per_element("num_nodes_per_element", num_elems),
          num_qps_per_element("num_qps_per_element", num_elems),
          node_state_indices("node_state_indices", num_elems, max_elem_nodes),
          element_freedom_signature("element_freedom_signature", num_elems, max_elem_nodes),
          element_freedom_table("element_freedom_table", num_elems, max_elem_nodes),
          gravity("gravity"),
          // Node Data
          node_x0("node_x0", num_elems, max_elem_nodes),
          node_u("node_u", num_elems, max_elem_nodes),
          node_u_dot("node_u_dot", num_elems, max_elem_nodes),
          node_u_ddot("node_u_ddot", num_elems, max_elem_nodes),
          node_FX("node_force_external", num_elems, max_elem_nodes),
          // Quadrature Point data
          qp_weight("qp_weight", num_elems, max_elem_qps),
          qp_jacobian("qp_jacobian", num_elems, max_elem_qps),
          qp_Mstar("qp_Mstar", num_elems, max_elem_qps),
          qp_Cstar("qp_Cstar", num_elems, max_elem_qps),
          qp_x("qp_x", num_elems, max_elem_qps),
          qp_x0("qp_x0", num_elems, max_elem_qps),
          qp_x0_prime("qp_x0_prime", num_elems, max_elem_qps),
          qp_r0("qp_r0", num_elems, max_elem_qps),
          qp_u("qp_u", num_elems, max_elem_qps),
          qp_u_prime("qp_u_prime", num_elems, max_elem_qps),
          qp_u_dot("qp_u_dot", num_elems, max_elem_qps),
          qp_u_ddot("qp_u_ddot", num_elems, max_elem_qps),
          qp_r("qp_r", num_elems, max_elem_qps),
          qp_r_prime("qp_r_prime", num_elems, max_elem_qps),
          qp_omega("qp_omega", num_elems, max_elem_qps),
          qp_omega_dot("qp_omega_dot", num_elems, max_elem_qps),
          qp_deformation("qp_deformation", num_elems, max_elem_qps),
          qp_E("qp_E", num_elems, max_elem_qps),
          qp_Fe("qp_Fe", num_elems, max_elem_qps),
          residual_vector_terms("residual_vector_terms", num_elems, max_elem_nodes),
          system_matrix_terms("system_matrix_terms", num_elems, max_elem_nodes, max_elem_nodes),
          // Shape Function data
          shape_interp("shape_interp", num_elems, max_elem_nodes, max_elem_qps),
          shape_deriv("deriv_interp", num_elems, max_elem_nodes, max_elem_qps) {
        Kokkos::deep_copy(element_freedom_signature, FreedomSignature::AllComponents);
    }
};

}  // namespace openturbine
