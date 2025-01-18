#pragma once

#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"
#include "types.hpp"

namespace openturbine {

/**
 * @brief Contains field variables for spring elements to compute per-element
 * contributions to the residual vector and system/iteration matrix
 */
struct Springs {
    size_t num_elems;  //< Total number of elements

    Kokkos::View<size_t*> num_nodes_per_element;  //< This is always 2 for springs
    Kokkos::View<size_t* [2]> node_state_indices;
    Kokkos::View<FreedomSignature* [2]> element_freedom_signature;
    Kokkos::View<size_t* [2][3]> element_freedom_table;  //< Only translational DOFs for springs

    Kokkos::View<double* [3]> x0;          //< Initial distance vector between nodes
    Kokkos::View<double* [3]> u1;          //< Displacement of node 1
    Kokkos::View<double* [3]> u2;          //< Displacement of node 2
    Kokkos::View<double* [3]> r;           //< Current distance vector between nodes
    Kokkos::View<double*> l;               //< Current length of springs
    Kokkos::View<double*> l_ref;           //< Initial length of springs
    Kokkos::View<double*> k;               //< Spring stiffness coefficients
    Kokkos::View<double*> c1;              //< First coefficient for force calculation
    Kokkos::View<double*> c2;              //< Second coefficient for force calculation
    Kokkos::View<double* [3]> f;           //< Force components
    Kokkos::View<double* [3][3]> a;        //< Stiffness matrices
    Kokkos::View<double* [3][3]> r_tilde;  //< Skew-symmetric matrix of r

    Kokkos::View<double* [2][3]> residual_vector_terms;
    Kokkos::View<double* [2][2][3][3]> stiffness_matrix_terms;

    explicit Springs(const size_t n_spring_elems)
        : num_elems(n_spring_elems),
          num_nodes_per_element("num_nodes_per_element", num_elems),
          node_state_indices("node_state_indices", num_elems),
          element_freedom_signature("element_freedom_signature", num_elems),
          element_freedom_table("element_freedom_table", num_elems),
          x0("x0", num_elems),
          u1("u1", num_elems),
          u2("u2", num_elems),
          r("r", num_elems),
          l("l", num_elems),
          l_ref("l_ref", num_elems),
          k("k", num_elems),
          c1("c1", num_elems),
          c2("c2", num_elems),
          f("f", num_elems),
          a("a", num_elems),
          r_tilde("r_tilde", num_elems),
          residual_vector_terms("residual_vector_terms", num_elems),
          stiffness_matrix_terms("stiffness_matrix_terms", num_elems) {
        Kokkos::deep_copy(num_nodes_per_element, 2);  // Always 2 nodes per element
        Kokkos::deep_copy(
            element_freedom_signature, FreedomSignature::JustPosition
        );  // Springs only have translational DOFs
    }
};

}  // namespace openturbine
