#include <Kokkos_Core.hpp>

#include "src/restruct_poc/beams/beams.hpp"

namespace openturbine {
struct ComputeNumberOfNonZeros {
    Kokkos::View<Beams::ElemIndices*>::const_type elem_indices;

    KOKKOS_FUNCTION
    void operator()(int i_elem, int& update) const {
        auto idx = elem_indices[i_elem];
        auto num_nodes = idx.num_nodes;
        update += (num_nodes * 6) * (num_nodes * 6);
    }
};
}  // namespace openturbine