#pragma once

#include <numeric>

#include <KokkosBlas.hpp>

#include "types.hpp"

namespace openturbine {

struct Beams {
    // Node and quadrature point index data for an element
    struct ElemIndices {
        size_t num_nodes;
        size_t num_qps;
        Kokkos::pair<size_t, size_t> node_range;
        Kokkos::pair<size_t, size_t> qp_range;
        Kokkos::pair<size_t, size_t> qp_shape_range;
        ElemIndices() {}
        ElemIndices(size_t num_nodes_, size_t num_qps_, size_t i_node_start, size_t i_qp_start)
            : num_nodes(num_nodes_),
              num_qps(num_qps_),
              node_range(Kokkos::make_pair(i_node_start, i_node_start + num_nodes)),
              qp_range(Kokkos::make_pair(i_qp_start, i_qp_start + num_qps)),
              qp_shape_range(Kokkos::make_pair((size_t)0, num_qps)) {}
    };

    size_t num_elems;  // Number of beams
    size_t num_nodes;  // Number of nodes
    size_t num_qps;    // Number of quadrature points

    Kokkos::View<ElemIndices*> elem_indices;   // View of element node and qp indices into views
    Kokkos::View<size_t*> node_state_indices;  // State row index for each node

    View_3 gravity;

    // Node-based data
    View_Nx7 node_x0;         // Inital position/rotation
    View_Nx7 node_u;          // State: translation/rotation displacement
    View_Nx6 node_u_dot;      // State: translation/rotation velocity
    View_Nx6 node_u_ddot;     // State: translation/rotation acceleration
    View_Nx6 node_FE;         // Elastic forces
    View_Nx6 node_FG;         // Gravity forces
    View_Nx6 node_FI;         // Inertial forces
    View_Nx6 node_FX;         // External forces
    View_N node_residual;     // Residual forces
    View_NxN node_iteration;  // Iteration matrix

    // Quadrature point data
    View_N qp_weight;       // Integration weights
    View_N qp_jacobian;     // Jacobian vector
    View_Nx6x6 qp_Mstar;    // Mass matrix in material frame
    View_Nx6x6 qp_Cstar;    // Stiffness matrix in material frame
    View_Nx3 qp_x0;         // Initial position
    View_Nx3 qp_x0_prime;   // Initial position derivative
    View_Nx4 qp_r0;         // Initial rotation
    View_Nx3 qp_u;          // State: translation displacement
    View_Nx3 qp_u_prime;    // State: translation displacement derivative
    View_Nx3 qp_u_dot;      // State: translation velocity
    View_Nx3 qp_u_ddot;     // State: translation acceleration
    View_Nx4 qp_r;          // State: rotation
    View_Nx4 qp_r_prime;    // State: rotation derivative
    View_Nx3 qp_omega;      // State: angular velocity
    View_Nx3 qp_omega_dot;  // State: position/rotation
    View_Nx6 qp_strain;     // Strain
    View_Nx6 qp_Fc;         // Elastic force
    View_Nx6 qp_Fd;         // Elastic force
    View_Nx6 qp_Fi;         // Inertial force
    View_Nx6 qp_Fg;         // Gravity force
    View_Nx6x6 qp_RR0;      // Global rotation
    View_Nx6x6 qp_Muu;      // Mass in global frame
    View_Nx6x6 qp_Cuu;      // Stiffness in global frame
    View_Nx6x6 qp_Ouu;      // Linearization matrices
    View_Nx6x6 qp_Puu;      // Linearization matrices
    View_Nx6x6 qp_Quu;      // Linearization matrices
    View_Nx6x6 qp_Guu;      // Linearization matrices
    View_Nx6x6 qp_Kuu;      // Linearization matrices

    View_NxN shape_interp;  // shape function matrix for interpolation [Nodes x QPs]
    View_NxN shape_deriv;   // shape function matrix for derivative interp [Nodes x QPs]

    // Scratch variables to be replaced later
    View_Nx6x6 M_6x6;   //
    View_Nx3x4 M_3x4;   //
    View_Nx3x3 M1_3x3;  //
    View_Nx3x3 M2_3x3;  //
    View_Nx3x3 M3_3x3;  //
    View_Nx3x3 M4_3x3;  //
    View_Nx3x3 M5_3x3;  //
    View_Nx3x3 M6_3x3;  //
    View_Nx3x3 M7_3x3;  //
    View_Nx3x3 M8_3x3;  //
    View_Nx3x3 M9_3x3;  //
    View_Nx3 V1_3;      //
    View_Nx3 V2_3;      //
    View_Nx3 V3_3;      //
    View_Nx4 qp_quat;   //

    Beams() {}  // Default constructor which doesn't initialize views

    // Constructor which initializes views based on given sizes
    Beams(
        const size_t num_beams_, const size_t num_nodes_, const size_t num_qps_,
        const size_t max_elem_qps
    )
        : num_elems(num_beams_),
          num_nodes(num_nodes_),
          num_qps(num_qps_),
          // Element Data
          elem_indices("elem_indices", num_beams_),
          node_state_indices("node_state_indices", num_nodes_),
          gravity("gravity"),
          // Node Data
          node_x0("node_x0", num_nodes_),
          node_u("node_u", num_nodes_),
          node_u_dot("node_u_dot", num_nodes_),
          node_u_ddot("node_u_ddot", num_nodes_),
          node_FE("node_force_elastic", num_nodes_),
          node_FG("node_force_gravity", num_nodes_),
          node_FI("node_force_inertial", num_nodes_),
          node_FX("node_force_external", num_nodes_),
          node_residual("node_residual", num_nodes_ * 6),
          node_iteration("node_iteration", num_nodes_ * 6, num_nodes_ * 6),
          // Quadrature Point data
          qp_weight("qp_weight", num_qps_),
          qp_jacobian("qp_jacobian", num_qps_),
          qp_Mstar("qp_Mstar", num_qps_),
          qp_Cstar("qp_Cstar", num_qps_),
          qp_x0("qp_x0", num_qps_),
          qp_x0_prime("qp_x0_prime", num_qps_),
          qp_r0("qp_r0", num_qps_),
          qp_u("qp_u", num_qps_),
          qp_u_prime("qp_u_prime", num_qps_),
          qp_u_dot("qp_u_dot", num_qps_),
          qp_u_ddot("qp_u_ddot", num_qps_),
          qp_r("qp_r", num_qps_),
          qp_r_prime("qp_r_prime", num_qps_),
          qp_omega("qp_omega", num_qps_),
          qp_omega_dot("qp_omega_dot", num_qps_),
          qp_strain("qp_strain", num_qps_),
          qp_Fc("qp_Fc", num_qps_),
          qp_Fd("qp_Fd", num_qps_),
          qp_Fi("qp_Fi", num_qps_),
          qp_Fg("qp_Fg", num_qps_),
          qp_RR0("qp_RR0", num_qps_),
          qp_Muu("qp_Muu", num_qps_),
          qp_Cuu("qp_Cuu", num_qps_),
          qp_Ouu("qp_Ouu", num_qps_),
          qp_Puu("qp_Puu", num_qps_),
          qp_Quu("qp_Quu", num_qps_),
          qp_Guu("qp_Guu", num_qps_),
          qp_Kuu("qp_Kuu", num_qps_),
          shape_interp("shape_interp", num_nodes_, max_elem_qps),
          shape_deriv("deriv_interp", num_nodes_, max_elem_qps),
          // Scratch
          M_6x6("M1_6x6", num_qps_),
          M_3x4("M_3x4", num_qps_),
          M1_3x3("R1_3x3", num_qps_),
          M2_3x3("R1_3x3", num_qps_),
          M3_3x3("R1_3x3", num_qps_),
          M4_3x3("R1_3x3", num_qps_),
          M5_3x3("R1_3x3", num_qps_),
          M6_3x3("R1_3x3", num_qps_),
          M7_3x3("R1_3x3", num_qps_),
          M8_3x3("R1_3x3", num_qps_),
          M9_3x3("R1_3x3", num_qps_),
          V1_3("V_3", num_qps_),
          V2_3("V_3", num_qps_),
          V3_3("V_3", num_qps_),
          qp_quat("Quat_4", num_qps_) {}
};

}  // namespace openturbine