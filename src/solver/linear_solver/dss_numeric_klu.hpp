#pragma once

#include <Kokkos_Core.hpp>
#include <klu.h>

#include "dss_algorithm.hpp"
#include "dss_handle_klu.hpp"

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
            klu_refactor(
                const_cast<int*>(row_ptrs.data()), col_inds.data(), values.data(), symbolic, numeric,
                &common
            );
        } else {
            numeric = klu_factor(
                const_cast<int*>(row_ptrs.data()), col_inds.data(), values.data(), symbolic, &common
            );
        }
    }
};

}  // namespace openturbine
