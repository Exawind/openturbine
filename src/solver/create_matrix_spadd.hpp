#pragma once

#include <KokkosSparse.hpp>
#include <Kokkos_Core.hpp>

namespace openturbine {

template <typename CrsMatrixType, typename KernelHandle>
[[nodiscard]] inline CrsMatrixType CreateMatrixSpadd(
    const CrsMatrixType& A, const CrsMatrixType& B, KernelHandle& handle
) {
    auto C = CrsMatrixType{};
    handle.create_spadd_handle(true, true);
    KokkosSparse::spadd_symbolic(&handle, A, B, C);
    KokkosSparse::spadd_numeric(&handle, 1., A, 1., B, C);
    return C;
}

}  // namespace openturbine
