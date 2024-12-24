#pragma once

#include <Amesos2.hpp>
#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename GlobalCrsMatrixType, typename GlobalMultiVectorType>
[[nodiscard]] inline Teuchos::RCP<GlobalCrsMatrixType> CreateGlobalMatrix(
    const typename GlobalCrsMatrixType::local_matrix_device_type& full_matrix
) {
    using CrsMatrixType = typename GlobalCrsMatrixType::local_matrix_device_type;
    using LocalOrdinalType = typename GlobalCrsMatrixType::local_ordinal_type;
    using GlobalOrdinalType = typename GlobalCrsMatrixType::global_ordinal_type;
    auto comm = Teuchos::createSerialComm<LocalOrdinalType>();
    auto row_map = Tpetra::createLocalMap<LocalOrdinalType, GlobalOrdinalType>(
        static_cast<size_t>(full_matrix.numRows()), comm
    );
    auto col_map = Tpetra::createLocalMap<LocalOrdinalType, GlobalOrdinalType>(
        static_cast<size_t>(full_matrix.numCols()), comm
    );

    return Teuchos::make_rcp<GlobalCrsMatrixType>(row_map, col_map, CrsMatrixType("A", full_matrix));
}

}  // namespace openturbine
