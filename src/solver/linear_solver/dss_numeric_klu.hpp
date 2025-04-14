#pragma once

#include <klu.h>

namespace openturbine {
template <typename CrsMatrixType>
struct DSSNumericFunction<DSSHandle<DSSAlgorithm::KLU>, CrsMatrixType> {
    static void numeric(DSSHandle<DSSAlgorithm::KLU>& dss_handle, CrsMatrixType& A) {
        auto* values = A.values.data();
        auto* row_ptrs = A.graph.row_map.data();
        auto* col_inds = A.graph.entries.data();

        auto*& symbolic = dss_handle.get_symbolic();
        auto*& numeric = dss_handle.get_numeric();
        auto& common = dss_handle.get_common();

        if (numeric != nullptr) {
            klu_free_numeric(&numeric, &common);
        }
        numeric = klu_factor(const_cast<int*>(row_ptrs), col_inds, values, symbolic, &common);
    }
};

}  // namespace openturbine
