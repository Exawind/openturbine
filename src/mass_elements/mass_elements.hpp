#pragma once

#include <Kokkos_Core.hpp>

#include "src/dof_management/freedom_signature.hpp"
#include "src/types.hpp"

namespace openturbine {

/**
 * @brief Data structure containing field variables for element-level computations of mass elements.
 *
 * This struct holds all the necessary field variables needed to compute per-element
 * contributions to the residual vector and system matrix for mass-only/rigid body elements.
 *
 * @details The data is organized using Kokkos::View containers with the following conventions:
 * - All Views use element index as their first dimension
 * - Views prefixed with `node_` use node index as their second dimension
 * - Views prefixed with `qp_` use quadrature point index as their second dimension
 * - Additional dimensions represent physical components (e.g., xyz coordinates, rotations)
 *
 * The struct contains three main categories of data:
 * 1. Element metadata (sizes, indices, freedom signatures)
 * 2. Node-based quantities (positions, displacements, velocities, accelerations)
 * 3. Quadrature point quantities (mass matrices, forces, kinematic variables)
 */
struct MassElements {
    // Metadata for mass elements in the mesh
    size_t num_elems;       //< Total number of elements
    size_t max_elem_nodes;  //< Maximum number of nodes per element
    size_t max_elem_qps;    //< Maximum number of quadrature points per element

    Kokkos::View<size_t*> num_nodes_per_element;
    Kokkos::View<size_t*> num_qps_per_element;
    Kokkos::View<size_t**> node_state_indices;  //< State row index for each node
    Kokkos::View<FreedomSignature**> element_freedom_signature;
    Kokkos::View<size_t* [6]> element_freedom_table;

    View_3 gravity;

    // Node-based data
    Kokkos::View<double* [7]> node_x0;      //< Initial position/rotation
    Kokkos::View<double* [7]> node_u;       //< State: translation/rotation displacement
    Kokkos::View<double* [6]> node_u_dot;   //< State: translation/rotation velocity
    Kokkos::View<double* [6]> node_u_ddot;  //< State: translation/rotation acceleration

    // Quadrature point data
    Kokkos::View<double* [6][6]> qp_Mstar;            //< Mass matrix in material frame
    Kokkos::View<double* [7]> qp_x;                   //< Current position/orientation
    Kokkos::View<double* [3]> qp_x0;                  //< Initial position
    Kokkos::View<double* [4]> qp_r0;                  //< Initial rotation
    Kokkos::View<double* [3]> qp_u;                   //< State: translation displacement
    Kokkos::View<double* [3]> qp_u_dot;               //< State: translation velocity
    Kokkos::View<double* [3]> qp_u_ddot;              //< State: translation acceleration
    Kokkos::View<double* [4]> qp_r;                   //< State: rotation
    Kokkos::View<double* [3]> qp_omega;               //< State: angular velocity
    Kokkos::View<double* [3]> qp_omega_dot;           //< State: position/rotation
    Kokkos::View<double* [3][3]> qp_eta_tilde;        //
    Kokkos::View<double* [3][3]> qp_omega_tilde;      //
    Kokkos::View<double* [3][3]> qp_omega_dot_tilde;  //
    Kokkos::View<double* [3]> qp_eta;                 //
    Kokkos::View<double* [3][3]> qp_rho;              //
    Kokkos::View<double* [6]> qp_Fi;                  //< Inertial force
    Kokkos::View<double* [6]> qp_Fg;                  //< Gravity force
    Kokkos::View<double* [6][6]> qp_RR0;              //< Global rotation
    Kokkos::View<double* [6][6]> qp_Muu;              //< Mass matrix in global frame

    MassElements(const size_t n_mass_elems, const size_t max_e_nodes, const size_t max_e_qps)
        :  // Metadata
          num_elems(n_mass_elems),
          max_elem_nodes(max_e_nodes),
          max_elem_qps(max_e_qps),
          // Element data
          num_nodes_per_element("num_nodes_per_element", num_elems),
          num_qps_per_element("num_qps_per_element", num_elems),
          node_state_indices("node_state_indices", num_elems, max_elem_nodes),
          element_freedom_signature("element_freedom_signature", num_elems, max_elem_nodes),
          element_freedom_table("element_freedom_table", num_elems, max_elem_nodes),
          gravity("gravity"),
          // Node data
          node_x0("node_x0", num_elems, max_elem_nodes),
          node_u("node_u", num_elems, max_elem_nodes),
          node_u_dot("node_u_dot", num_elems, max_elem_nodes),
          node_u_ddot("node_u_ddot", num_elems, max_elem_nodes),
          // Quadrature point data
          qp_Mstar("qp_Mstar", num_elems, max_elem_qps),
          qp_x("qp_x", num_elems, max_elem_qps),
          qp_x0("qp_x0", num_elems, max_elem_qps),
          qp_r0("qp_r0", num_elems, max_elem_qps),
          qp_u("qp_u", num_elems, max_elem_qps),
          qp_u_dot("qp_u_dot", num_elems, max_elem_qps),
          qp_u_ddot("qp_u_ddot", num_elems, max_elem_qps),
          qp_r("qp_r", num_elems, max_elem_qps),
          qp_omega("qp_omega", num_elems, max_elem_qps),
          qp_omega_dot("qp_omega_dot", num_elems, max_elem_qps),
          qp_eta_tilde("R1_3x3", num_elems, max_elem_qps),
          qp_omega_tilde("R1_3x3", num_elems, max_elem_qps),
          qp_omega_dot_tilde("R1_3x3", num_elems, max_elem_qps),
          qp_eta("V_3", num_elems, max_elem_qps),
          qp_rho("R1_3x3", num_elems, max_elem_qps),
          qp_Fi("qp_Fi", num_elems, max_elem_qps),
          qp_Fg("qp_Fg", num_elems, max_elem_qps),
          qp_RR0("qp_RR0", num_elems, max_elem_qps),
          qp_Muu("qp_Muu", num_elems, max_elem_qps) {
        Kokkos::deep_copy(element_freedom_signature, FreedomSignature::AllComponents);
    }
};

}  // namespace openturbine
