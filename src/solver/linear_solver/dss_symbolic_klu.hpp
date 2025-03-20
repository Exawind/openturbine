#pragma once

#include <klu.h>

#include "dss_handle_klu.hpp"

namespace openturbine {
template <typename CrsMatrixType>
struct DSSSymbolicFunction<DSSHandle<DSSAlgorithm::KLU>, CrsMatrixType> {
    static void symbolic(DSSHandle<DSSAlgorithm::KLU>& dss_handle, CrsMatrixType& A) {
        const auto num_rows = A.numRows();

        auto* row_ptrs = A.graph.row_map.data();
        auto* col_inds = A.graph.entries.data();

        auto*& symbolic = dss_handle.get_symbolic();
        auto& common = dss_handle.get_common();
        if (symbolic != nullptr) {
            klu_free_symbolic(&symbolic, &common);
        }
        symbolic = klu_analyze(num_rows, const_cast<int*>(row_ptrs), col_inds, &common);
    }
};

}  // namespace openturbine
