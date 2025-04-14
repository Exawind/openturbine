#pragma once

#include <umfpack.h>

namespace openturbine {
template <typename CrsMatrixType>
struct DSSNumericFunction<DSSHandle<DSSAlgorithm::UMFPACK>, CrsMatrixType> {
    static void numeric(DSSHandle<DSSAlgorithm::UMFPACK>& dss_handle, CrsMatrixType& A) {
        auto* values = A.values.data();
        auto* row_ptrs = A.graph.row_map.data();
        auto* col_inds = A.graph.entries.data();

        auto*& symbolic = dss_handle.get_symbolic();
        auto*& numeric = dss_handle.get_numeric();

        umfpack_di_numeric(row_ptrs, col_inds, values, symbolic, &numeric, nullptr, nullptr);
    }
};

}  // namespace openturbine
