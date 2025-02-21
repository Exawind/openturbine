#pragma once

#include <Kokkos_Core.hpp>

#include "dof_management/freedom_signature.hpp"
#include "types.hpp"

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
    Kokkos::View<size_t*> state_indices;
    Kokkos::View<FreedomSignature*> element_freedom_signature;
    Kokkos::View<size_t* [6]> element_freedom_table;

    View_3 gravity;

    Kokkos::View<double* [7]> node_x0;      //< Initial position/rotation
    Kokkos::View<double* [6][6]> qp_Mstar;  //< Mass matrix in material frame

    Kokkos::View<double* [6]> residual_vector_terms;
    Kokkos::View<double* [6][6]> system_matrix_terms;

    explicit Masses(const size_t n_mass_elems)
        : num_elems(n_mass_elems),
          num_nodes_per_element("num_nodes_per_element", num_elems),
          state_indices("state_indices", num_elems),
          element_freedom_signature("element_freedom_signature", num_elems),
          element_freedom_table("element_freedom_table", num_elems),
          gravity("gravity"),
          node_x0("node_x0", num_elems),
          qp_Mstar("qp_Mstar", num_elems),
          residual_vector_terms("residual_vector_terms", num_elems),
          system_matrix_terms("system_matrix_terms", num_elems) {
        Kokkos::deep_copy(num_nodes_per_element, 1);  // Always 1 node per element
        Kokkos::deep_copy(element_freedom_signature, FreedomSignature::AllComponents);
    }
};

}  // namespace openturbine
