#pragma once

#include <Kokkos_Core.hpp>
#include "src/restruct_poc/types.hpp"

namespace openturbine {

struct InterpolateQPVector {
    int first_qp;
    int first_node;
    int num_nodes;
    View_NxN::const_type shape_interp;                           
    View_Nx3::const_type node_vector;  
    View_Nx3 qp_vector;    

    KOKKOS_FUNCTION
    void operator()(int j_index) const {
        const auto j = first_qp + j_index;
        auto local_total = Kokkos::Array<double, 3>{};
        for (int i_index = 0; i_index < num_nodes; ++i_index) {
            const auto i = first_node + i_index;
            const auto phi = shape_interp(i, j_index);
            for (int k = 0; k < 3; ++k) {
                local_total[k] += node_vector(i, k) * phi;
            }
        }
        for (int k = 0; k < 3; ++k) {
            qp_vector(j, k) = local_total[k];
        }
    }
};

}