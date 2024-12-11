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

    Kokkos::View<double* [1][7]> x0;        //< Initial position/rotation
    Kokkos::View<double* [1][7]> u;         //< State: translation/rotation displacement
    Kokkos::View<double* [1][6]> u_dot;     //< State: translation/rotation velocity
    Kokkos::View<double* [1][6]> u_ddot;    //< State: translation/rotation acceleration
    Kokkos::View<double* [1][7]> x;         //< Current position/orientation
    Kokkos::View<double* [1][6][6]> Mstar;  //< Mass matrix in material frame
    Kokkos::View<double* [1][3]> eta;       //< Offset between mass center and elastic axis
    Kokkos::View<double* [1][3][3]> rho;    //< Rotational inertia part of mass matrix
    Kokkos::View<double* [1][6]> Fi;        //< Inertial force
    Kokkos::View<double* [1][6]> Fg;        //< Gravity force
    Kokkos::View<double* [1][6][6]> RR0;    //< Global rotation
    Kokkos::View<double* [1][6][6]> Muu;    //< Mass matrix in global frame

    Masses(const size_t n_mass_elems)
        : num_elems(n_mass_elems),
          num_nodes_per_element("num_nodes_per_element", num_elems),
          state_indices("state_indices", num_elems),
          element_freedom_signature("element_freedom_signature", num_elems),
          element_freedom_table("element_freedom_table", num_elems),
          gravity("gravity"),
          x0("x0", num_elems),
          u("u", num_elems),
          u_dot("u_dot", num_elems),
          u_ddot("u_ddot", num_elems),
          x("x", num_elems),
          Mstar("Mstar", num_elems),
          eta("eta", num_elems),
          rho("rho", num_elems),
          Fi("Fi", num_elems),
          Fg("Fg", num_elems),
          RR0("RR0", num_elems),
          Muu("Muu", num_elems) {
        Kokkos::deep_copy(num_nodes_per_element, 1);  // Always 1 node per element
        Kokkos::deep_copy(element_freedom_signature, FreedomSignature::AllComponents);
    }
};

}  // namespace openturbine
