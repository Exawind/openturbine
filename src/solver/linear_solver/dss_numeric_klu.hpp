#pragma once

#include <klu.h>

namespace openturbine {
template <typename CrsMatrixType>
struct DSSNumericFunction<DSSHandle<DSSAlgorithm::KLU>, CrsMatrixType> {
    static void numeric(DSSHandle<DSSAlgorithm::KLU>& dss_handle, CrsMatrixType& A) {
        auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.values);
        auto row_ptrs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.graph.row_map);
        auto col_inds = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.graph.entries);

        auto*& symbolic = dss_handle.get_symbolic();
        auto*& numeric = dss_handle.get_numeric();
        auto& common = dss_handle.get_common();

        if (numeric != nullptr) {
            klu_free_numeric(&numeric, &common);
        }
        numeric = klu_factor(
            const_cast<int*>(row_ptrs.data()), col_inds.data(), values.data(), symbolic, &common
        );
    }
};

}  // namespace openturbine
