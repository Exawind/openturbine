#pragma once

#include <Kokkos_Core.hpp>

#include "src/dof_management/freedom_signature.hpp"
#include "src/types.hpp"

namespace openturbine {

/**
 * @brief Contains field variables for mass elements (aka, rigid bodies) to compute per-element
 * contributions to the residual vector and system/iteration matrix
 *
 * @note Mass elements consist of a single node and no quadrature points are required, hence no
 * node/qp prefix is used in the Views (as opposed to Beams)
 */
struct Masses {
    size_t num_elems;  //< Total number of elements

    Kokkos::View<size_t*> num_nodes_per_element;  //< This is always 1 for masses
    Kokkos::View<size_t* [1]> state_indices;
    Kokkos::View<FreedomSignature* [1]> element_freedom_signature;
    Kokkos::View<size_t* [1][6]> element_freedom_table;

    View_3 gravity;

    Kokkos::View<double* [1][7]> node_x0;      //< Initial position/rotation
    Kokkos::View<double* [1][7]> node_u;       //< State: translation/rotation displacement
    Kokkos::View<double* [1][6]> node_u_dot;   //< State: translation/rotation velocity
    Kokkos::View<double* [1][6]> node_u_ddot;  //< State: translation/rotation acceleration

    Kokkos::View<double* [1][6][6]> qp_Mstar;  //< Mass matrix in material frame
    Kokkos::View<double* [1][7]> qp_x;         //< Current position/orientation
    Kokkos::View<double* [1][3]> qp_x0;
    Kokkos::View<double* [1][4]> qp_r0;
    Kokkos::View<double* [1][3]> qp_u;
    Kokkos::View<double* [1][3]> qp_u_ddot;
    Kokkos::View<double* [1][4]> qp_r;
    Kokkos::View<double* [1][3]> qp_omega;
    Kokkos::View<double* [1][3]> qp_omega_dot;
    Kokkos::View<double* [1][3][3]> qp_eta_tilde;
    Kokkos::View<double* [1][3][3]> qp_omega_tilde;
    Kokkos::View<double* [1][3][3]> qp_omega_dot_tilde;
    Kokkos::View<double* [1][3]> qp_eta;     //< Offset between mass center and elastic axis
    Kokkos::View<double* [1][3][3]> qp_rho;  //< Rotational inertia part of mass matrix
    Kokkos::View<double* [1][6]> qp_Fi;      //< Inertial force
    Kokkos::View<double* [1][6]> qp_Fg;      //< Gravity force
    Kokkos::View<double* [1][6][6]> qp_RR0;  //< Global rotation
    Kokkos::View<double* [1][6][6]> qp_Muu;  //< Mass matrix in global/inertial frame
    Kokkos::View<double* [1][6][6]> qp_Guu;  //< Gyroscopic/inertial damping matrix
    Kokkos::View<double* [1][6][6]> qp_Kuu;  //< Inertia stiffness matrix

    Kokkos::View<double* [1][6]> residual_vector_terms;
    Kokkos::View<double* [1][1][6][6]> stiffness_matrix_terms;
    Kokkos::View<double* [1][1][6][6]> inertia_matrix_terms;

    Masses(const size_t n_mass_elems)
        : num_elems(n_mass_elems),
          num_nodes_per_element("num_nodes_per_element", num_elems),
          state_indices("state_indices", num_elems),
          element_freedom_signature("element_freedom_signature", num_elems),
          element_freedom_table("element_freedom_table", num_elems),
          gravity("gravity"),
          node_x0("node_x0", num_elems),
          node_u("node_u", num_elems),
          node_u_dot("node_u_dot", num_elems),
          node_u_ddot("node_u_ddot", num_elems),
          qp_Mstar("qp_Mstar", num_elems),
          qp_x("qp_x", num_elems),
          qp_x0("qp_x0", num_elems),
          qp_r0("qp_r0", num_elems),
          qp_u("qp_u", num_elems),
          qp_u_ddot("qp_u_ddot", num_elems),
          qp_r("qp_r", num_elems),
          qp_omega("qp_omega", num_elems),
          qp_omega_dot("qp_omega_dot", num_elems),
          qp_eta_tilde("qp_eta_tilde", num_elems),
          qp_omega_tilde("qp_omega_tilde", num_elems),
          qp_omega_dot_tilde("qp_omega_dot_tilde", num_elems),
          qp_eta("qp_eta", num_elems),
          qp_rho("qp_rho", num_elems),
          qp_Fi("qp_Fi", num_elems),
          qp_Fg("qp_Fg", num_elems),
          qp_RR0("qp_RR0", num_elems),
          qp_Muu("qp_Muu", num_elems),
          qp_Guu("qp_Guu", num_elems),
          qp_Kuu("qp_Kuu", num_elems),
          residual_vector_terms("residual_vector_terms", num_elems),
          stiffness_matrix_terms("stiffness_matrix_terms", num_elems),
          inertia_matrix_terms("inertia_matrix_terms", num_elems) {
        Kokkos::deep_copy(num_nodes_per_element, 1);  // Always 1 node per element
        Kokkos::deep_copy(element_freedom_signature, FreedomSignature::AllComponents);
    }
};

}  // namespace openturbine
