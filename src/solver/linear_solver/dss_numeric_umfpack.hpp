#pragma once

#include <umfpack.h>

namespace openturbine {
template <typename CrsMatrixType>
struct DSSNumericFunction<DSSHandle<DSSAlgorithm::UMFPACK>, CrsMatrixType> {
    static void numeric(DSSHandle<DSSAlgorithm::UMFPACK>& dss_handle, CrsMatrixType& A) {
        auto values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.values);
        auto row_ptrs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.graph.row_map);
        auto col_inds = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.graph.entries);

        auto*& symbolic = dss_handle.get_symbolic();
        auto*& numeric = dss_handle.get_numeric();

        umfpack_di_numeric(
            row_ptrs.data(), col_inds.data(), values.data(), symbolic, &numeric, nullptr, nullptr
        );
    }
};

}  // namespace openturbine
