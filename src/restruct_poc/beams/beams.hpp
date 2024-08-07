#pragma once

#include <numeric>

#include <KokkosBlas.hpp>

#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct Beams {
    // Node and quadrature point index data for an element
    struct ElemIndices {
        size_t num_nodes;
        size_t num_qps;
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
    View_N qp_weight;               // Integration weights
    View_N qp_jacobian;             // Jacobian vector
    View_Nx6x6 qp_Mstar;            // Mass matrix in material frame
    View_Nx6x6 qp_Cstar;            // Stiffness matrix in material frame
    View_Nx3 qp_x0;                 // Initial position
    View_Nx3 qp_x0_prime;           // Initial position derivative
    View_Nx4 qp_r0;                 // Initial rotation
    View_Nx3 qp_u;                  // State: translation displacement
    View_Nx3 qp_u_prime;            // State: translation displacement derivative
    View_Nx3 qp_u_dot;              // State: translation velocity
    View_Nx3 qp_u_ddot;             // State: translation acceleration
    View_Nx4 qp_r;                  // State: rotation
    View_Nx4 qp_r_prime;            // State: rotation derivative
    View_Nx3 qp_omega;              // State: angular velocity
    View_Nx3 qp_omega_dot;          // State: position/rotation
    View_Nx3x4 qp_E;                // Quaternion derivative
    View_Nx3x3 qp_eta_tilde;        //
    View_Nx3x3 qp_omega_tilde;      //
    View_Nx3x3 qp_omega_dot_tilde;  //
    View_Nx3x3 qp_x0pupss;          //
    View_Nx3x3 qp_M_tilde;          //
    View_Nx3x3 qp_N_tilde;          //
    View_Nx3 qp_eta;                //
    View_Nx3x3 qp_rho;              //
    View_Nx6 qp_strain;             // Strain
    View_Nx6 qp_Fc;                 // Elastic force
    View_Nx6 qp_Fd;                 // Elastic force
    View_Nx6 qp_Fi;                 // Inertial force
    View_Nx6 qp_Fg;                 // Gravity force
    View_Nx6x6 qp_RR0;              // Global rotation
    View_Nx6x6 qp_Muu;              // Mass in global frame
    View_Nx6x6 qp_Cuu;              // Stiffness in global frame
    View_Nx6x6 qp_Ouu;              // Linearization matrices
    View_Nx6x6 qp_Puu;              // Linearization matrices
    View_Nx6x6 qp_Quu;              // Linearization matrices
    View_Nx6x6 qp_Guu;              // Linearization matrices
    View_Nx6x6 qp_Kuu;              // Linearization matrices

    View_NxN shape_interp;  // shape function matrix for interpolation [Nodes x QPs]
    View_NxN shape_deriv;   // shape function matrix for derivative interp [Nodes x QPs]

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
          qp_weight("qp_weight", n_qps),
          qp_jacobian("qp_jacobian", n_qps),
          qp_Mstar("qp_Mstar", n_qps),
          qp_Cstar("qp_Cstar", n_qps),
          qp_x0("qp_x0", n_qps),
          qp_x0_prime("qp_x0_prime", n_qps),
          qp_r0("qp_r0", n_qps),
          qp_u("qp_u", n_qps),
          qp_u_prime("qp_u_prime", n_qps),
          qp_u_dot("qp_u_dot", n_qps),
          qp_u_ddot("qp_u_ddot", n_qps),
          qp_r("qp_r", n_qps),
          qp_r_prime("qp_r_prime", n_qps),
          qp_omega("qp_omega", n_qps),
          qp_omega_dot("qp_omega_dot", n_qps),
          qp_E("qp_E", n_qps),
          qp_eta_tilde("R1_3x3", n_qps),
          qp_omega_tilde("R1_3x3", n_qps),
          qp_omega_dot_tilde("R1_3x3", n_qps),
          qp_x0pupss("R1_3x3", n_qps),
          qp_M_tilde("R1_3x3", n_qps),
          qp_N_tilde("R1_3x3", n_qps),
          qp_eta("V_3", n_qps),
          qp_rho("R1_3x3", n_qps),
          qp_strain("qp_strain", n_qps),
          qp_Fc("qp_Fc", n_qps),
          qp_Fd("qp_Fd", n_qps),
          qp_Fi("qp_Fi", n_qps),
          qp_Fg("qp_Fg", n_qps),
          qp_RR0("qp_RR0", n_qps),
          qp_Muu("qp_Muu", n_qps),
          qp_Cuu("qp_Cuu", n_qps),
          qp_Ouu("qp_Ouu", n_qps),
          qp_Puu("qp_Puu", n_qps),
          qp_Quu("qp_Quu", n_qps),
          qp_Guu("qp_Guu", n_qps),
          qp_Kuu("qp_Kuu", n_qps),
          shape_interp("shape_interp", n_nodes, max_e_qps),
          shape_deriv("deriv_interp", n_nodes, max_e_qps) {}
};

}  // namespace openturbine
