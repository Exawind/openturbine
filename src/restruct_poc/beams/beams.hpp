#pragma once

#include <numeric>

#include <KokkosBlas.hpp>

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct Beams {
    // Node and quadrature point index data for an element
    struct ElemIndices {
        size_t num_nodes{};
        size_t num_qps{};
        Kokkos::pair<size_t, size_t> node_range;
        Kokkos::pair<size_t, size_t> qp_range;
        Kokkos::pair<size_t, size_t> qp_shape_range;
        ElemIndices() = default;
        ElemIndices(size_t n_nodes, size_t n_qps, size_t i_node_start, size_t i_qp_start)
            : num_nodes(n_nodes),
              num_qps(n_qps),
              node_range(Kokkos::make_pair(i_node_start, i_node_start + n_nodes)),
              qp_range(Kokkos::make_pair(i_qp_start, i_qp_start + n_qps)),
              qp_shape_range(Kokkos::make_pair(0, n_qps)) {}
    };

    size_t num_elems;  // Number of beams
    size_t num_nodes;  // Number of nodes
    size_t num_qps;    // Number of quadrature points
    size_t max_elem_nodes;
    size_t max_elem_qps;

    Kokkos::View<ElemIndices*> elem_indices;   // View of element node and qp indices into views
    Kokkos::View<size_t*> node_state_indices;  // State row index for each node

    View_3 gravity;

    // Node-based data
    View_Nx7 node_x0;      // Inital position/rotation
    View_Nx7 node_u;       // State: translation/rotation displacement
    View_Nx6 node_u_dot;   // State: translation/rotation velocity
    View_Nx6 node_u_ddot;  // State: translation/rotation acceleration
    View_Nx6 node_FE;      // Elastic forces
    View_Nx6 node_FG;      // Gravity forces
    View_Nx6 node_FI;      // Inertial forces
    View_Nx6 node_FX;      // External forces

    // Quadrature point data
    View_NxN qp_weight;                                // Integration weights
    View_NxN qp_jacobian;                              // Jacobian vector
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
          elem_indices("elem_indices", n_beams),
          node_state_indices("node_state_indices", n_nodes),
          gravity("gravity"),
          // Node Data
          node_x0("node_x0", n_nodes),
          node_u("node_u", n_nodes),
          node_u_dot("node_u_dot", n_nodes),
          node_u_ddot("node_u_ddot", n_nodes),
          node_FE("node_force_elastic", n_nodes),
          node_FG("node_force_gravity", n_nodes),
          node_FI("node_force_inertial", n_nodes),
          node_FX("node_force_external", n_nodes),
          // Quadrature Point data
          qp_weight("qp_weight", n_beams, max_elem_qps),
          qp_jacobian("qp_jacobian", n_beams, max_elem_qps),
          qp_Mstar("qp_Mstar", n_beams, max_elem_qps),
          qp_Cstar("qp_Cstar", n_beams, max_elem_qps),
          qp_x0("qp_x0", n_beams, max_elem_qps),
          qp_x0_prime("qp_x0_prime", n_beams, max_elem_qps),
          qp_r0("qp_r0", n_beams, max_elem_qps),
          qp_u("qp_u", n_beams, max_elem_qps),
          qp_u_prime("qp_u_prime", n_beams, max_elem_qps),
          qp_u_dot("qp_u_dot", n_beams, max_elem_qps),
          qp_u_ddot("qp_u_ddot", n_beams, max_elem_qps),
          qp_r("qp_r", n_beams, max_elem_qps),
          qp_r_prime("qp_r_prime", n_beams, max_elem_qps),
          qp_omega("qp_omega", n_beams, max_elem_qps),
          qp_omega_dot("qp_omega_dot", n_beams, max_elem_qps),
          qp_E("qp_E", n_beams, max_elem_qps),
          qp_eta_tilde("R1_3x3", n_beams, max_elem_qps),
          qp_omega_tilde("R1_3x3", n_beams, max_elem_qps),
          qp_omega_dot_tilde("R1_3x3", n_beams, max_elem_qps),
          qp_x0pupss("R1_3x3", n_beams, max_elem_qps),
          qp_M_tilde("R1_3x3", n_beams, max_elem_qps),
          qp_N_tilde("R1_3x3", n_beams, max_elem_qps),
          qp_eta("V_3", n_beams, max_elem_qps),
          qp_rho("R1_3x3", n_beams, max_elem_qps),
          qp_strain("qp_strain", n_beams, max_elem_qps),
          qp_Fc("qp_Fc", n_beams, max_elem_qps),
          qp_Fd("qp_Fd", n_beams, max_elem_qps),
          qp_Fi("qp_Fi", n_beams, max_elem_qps),
          qp_Fg("qp_Fg", n_beams, max_elem_qps),
          qp_RR0("qp_RR0", n_beams, max_elem_qps),
          qp_Muu("qp_Muu", n_beams, max_elem_qps),
          qp_Cuu("qp_Cuu", n_beams, max_elem_qps),
          qp_Ouu("qp_Ouu", n_beams, max_elem_qps),
          qp_Puu("qp_Puu", n_beams, max_elem_qps),
          qp_Quu("qp_Quu", n_beams, max_elem_qps),
          qp_Guu("qp_Guu", n_beams, max_elem_qps),
          qp_Kuu("qp_Kuu", n_beams, max_elem_qps),
          shape_interp("shape_interp", n_beams, max_elem_nodes, max_elem_qps),
          shape_deriv("deriv_interp", n_beams, max_elem_nodes, max_elem_qps) {}
};

}  // namespace openturbine
