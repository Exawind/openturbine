#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename CrsMatrixType, typename KernelHandle>
[[nodiscard]] inline CrsMatrixType CreateMatrixSpgemm(
    const CrsMatrixType& A, const CrsMatrixType& B, KernelHandle& handle
) {
    auto C = CrsMatrixType{};
    handle.create_spgemm_handle(KokkosSparse::SPGEMMAlgorithm::SPGEMM_KK_DENSE);
    KokkosSparse::spgemm_symbolic(handle, A, false, B, false, C);
    KokkosSparse::spgemm_numeric(handle, A, false, B, false, C);
    return C;
}

}  // namespace openturbine
