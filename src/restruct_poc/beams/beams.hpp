#pragma once

#include <numeric>

#include <KokkosBlas.hpp>

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct Beams {
    size_t num_elems;  // Number of beams
    size_t num_nodes;  // Number of nodes
    size_t num_qps;    // Number of quadrature points
    size_t max_elem_nodes;
    size_t max_elem_qps;

    Kokkos::View<size_t*> num_nodes_per_element;
    Kokkos::View<size_t*> num_qps_per_element;
    Kokkos::View<size_t**> node_state_indices;  // State row index for each node

    View_3 gravity;

    // Node-based data
    Kokkos::View<double** [7]> node_x0;      // Inital position/rotation
    Kokkos::View<double** [7]> node_u;       // State: translation/rotation displacement
    Kokkos::View<double** [6]> node_u_dot;   // State: translation/rotation velocity
    Kokkos::View<double** [6]> node_u_ddot;  // State: translation/rotation acceleration
    Kokkos::View<double** [6]> node_FX;      // External forces

    // Quadrature point data
    Kokkos::View<double**> qp_weight;                  // Integration weights
    Kokkos::View<double**> qp_jacobian;                // Jacobian vector
    Kokkos::View<double** [6][6]> qp_Mstar;            // Mass matrix in material frame
    Kokkos::View<double** [6][6]> qp_Cstar;            // Stiffness matrix in material frame
    Kokkos::View<double** [3]> qp_x0;                  // Initial position
    Kokkos::View<double** [3]> qp_x0_prime;            // Initial position derivative
    Kokkos::View<double** [4]> qp_r0;                  // Initial rotation
    Kokkos::View<double** [3]> qp_u;                   // State: translation displacement
    Kokkos::View<double** [3]> qp_u_prime;             // State: translation displacement derivative
    Kokkos::View<double** [3]> qp_u_dot;               // State: translation velocity
    Kokkos::View<double** [3]> qp_u_ddot;              // State: translation acceleration
    Kokkos::View<double** [4]> qp_r;                   // State: rotation
    Kokkos::View<double** [4]> qp_r_prime;             // State: rotation derivative
    Kokkos::View<double** [3]> qp_omega;               // State: angular velocity
    Kokkos::View<double** [3]> qp_omega_dot;           // State: position/rotation
    Kokkos::View<double** [3][4]> qp_E;                // Quaternion derivative
    Kokkos::View<double** [3][3]> qp_eta_tilde;        //
    Kokkos::View<double** [3][3]> qp_omega_tilde;      //
    Kokkos::View<double** [3][3]> qp_omega_dot_tilde;  //
    Kokkos::View<double** [3][3]> qp_x0pupss;          //
    Kokkos::View<double** [3][3]> qp_M_tilde;          //
    Kokkos::View<double** [3][3]> qp_N_tilde;          //
    Kokkos::View<double** [3]> qp_eta;                 //
    Kokkos::View<double** [3][3]> qp_rho;              //
    Kokkos::View<double** [6]> qp_strain;              // Strain
    Kokkos::View<double** [6]> qp_Fc;                  // Elastic force
    Kokkos::View<double** [6]> qp_Fd;                  // Elastic force
    Kokkos::View<double** [6]> qp_Fi;                  // Inertial force
    Kokkos::View<double** [6]> qp_Fg;                  // Gravity force
    Kokkos::View<double** [6][6]> qp_RR0;              // Global rotation
    Kokkos::View<double** [6][6]> qp_Muu;              // Mass in global frame
    Kokkos::View<double** [6][6]> qp_Cuu;              // Stiffness in global frame
    Kokkos::View<double** [6][6]> qp_Ouu;              // Linearization matrices
    Kokkos::View<double** [6][6]> qp_Puu;              // Linearization matrices
    Kokkos::View<double** [6][6]> qp_Quu;              // Linearization matrices
    Kokkos::View<double** [6][6]> qp_Guu;              // Linearization matrices
    Kokkos::View<double** [6][6]> qp_Kuu;              // Linearization matrices

    Kokkos::View<double**[6]> residual_vector_terms;
    Kokkos::View<double***[6][6]> stiffness_matrix_terms;
    Kokkos::View<double***[6][6]> inertia_matrix_terms;

    Kokkos::View<double***> shape_interp;  // shape function matrix for interpolation [Nodes x QPs]
    Kokkos::View<double***>
        shape_deriv;  // shape function matrix for derivative interp [Nodes x QPs]

    // Constructor which initializes views based on given sizes
    Beams(
        const size_t n_beams, const size_t n_nodes, const size_t n_qps, const size_t max_e_nodes,
        const size_t max_e_qps
    )
        : num_elems(n_beams),
          num_nodes(n_nodes),
          num_qps(n_qps),
          max_elem_nodes(max_e_nodes),
          max_elem_qps(max_e_qps),
          // Element Data
          num_nodes_per_element("num_nodes_per_element", num_elems),
          num_qps_per_element("num_qps_per_element", num_elems),
          node_state_indices("node_state_indices", num_elems, max_elem_nodes),
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
          qp_E("qp_E", num_elems, max_elem_qps),
          qp_eta_tilde("R1_3x3", num_elems, max_elem_qps),
          qp_omega_tilde("R1_3x3", num_elems, max_elem_qps),
          qp_omega_dot_tilde("R1_3x3", num_elems, max_elem_qps),
          qp_x0pupss("R1_3x3", num_elems, max_elem_qps),
          qp_M_tilde("R1_3x3", num_elems, max_elem_qps),
          qp_N_tilde("R1_3x3", num_elems, max_elem_qps),
          qp_eta("V_3", num_elems, max_elem_qps),
          qp_rho("R1_3x3", num_elems, max_elem_qps),
          qp_strain("qp_strain", num_elems, max_elem_qps),
          qp_Fc("qp_Fc", num_elems, max_elem_qps),
          qp_Fd("qp_Fd", num_elems, max_elem_qps),
          qp_Fi("qp_Fi", num_elems, max_elem_qps),
          qp_Fg("qp_Fg", num_elems, max_elem_qps),
          qp_RR0("qp_RR0", num_elems, max_elem_qps),
          qp_Muu("qp_Muu", num_elems, max_elem_qps),
          qp_Cuu("qp_Cuu", num_elems, max_elem_qps),
          qp_Ouu("qp_Ouu", num_elems, max_elem_qps),
          qp_Puu("qp_Puu", num_elems, max_elem_qps),
          qp_Quu("qp_Quu", num_elems, max_elem_qps),
          qp_Guu("qp_Guu", num_elems, max_elem_qps),
          qp_Kuu("qp_Kuu", num_elems, max_elem_qps),
          residual_vector_terms("residual_vector_terms", num_elems, max_elem_nodes),
          stiffness_matrix_terms("stiffness_matrix_terms", num_elems, max_elem_nodes, max_elem_nodes),
          inertia_matrix_terms("inertia_matrix_terms", num_elems, max_elem_nodes, max_elem_nodes),
          shape_interp("shape_interp", num_elems, max_elem_nodes, max_elem_qps),
          shape_deriv("deriv_interp", num_elems, max_elem_nodes, max_elem_qps) {}
};

}  // namespace openturbine
