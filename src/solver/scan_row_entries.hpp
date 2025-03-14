#pragma once

#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename RowPtrType>
struct ScanRowEntries {
    using ValueType = typename RowPtrType::value_type;
    typename RowPtrType::const_type row_entries;
    RowPtrType row_ptrs;

    KOKKOS_FUNCTION
    void operator()(size_t i, ValueType& update, bool is_final) const {
        update += row_entries(i);
        if (is_final) {
            row_ptrs(i + 1) = update;
        }
    }
};

}
