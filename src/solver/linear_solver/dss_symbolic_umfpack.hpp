#pragma once

#include <umfpack.h>

#include "dss_handle_umfpack.hpp"

namespace openturbine {
template <typename CrsMatrixType>
struct DSSSymbolicFunction<DSSHandle<DSSAlgorithm::UMFPACK>, CrsMatrixType> {
    static void symbolic(DSSHandle<DSSAlgorithm::UMFPACK>& dss_handle, CrsMatrixType& A) {
        const auto num_rows = A.numRows();
        const auto num_cols = A.numCols();

        auto* row_ptrs = A.graph.row_map.data();
        auto* col_inds = A.graph.entries.data();

        auto*& symbolic = dss_handle.get_symbolic();

        umfpack_di_symbolic(
            num_rows, num_cols, row_ptrs, col_inds, nullptr, &symbolic, nullptr, nullptr
        );
    }
};
}  // namespace openturbine
