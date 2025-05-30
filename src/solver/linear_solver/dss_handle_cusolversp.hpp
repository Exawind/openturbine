#pragma once

#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

namespace openturbine {
template <>
class DSSHandle<DSSAlgorithm::CUSOLVER_SP> {
    struct cuSolverDssHandleType {
        cusolverSpHandle_t handle;
        cusparseMatDescr_t description;
        csrqrInfo_t info;
        Kokkos::View<char*> buffer;

        cuSolverDssHandleType() {
            cusolverSpCreate(&handle);
            cusparseCreateMatDescr(&description);
            cusparseSetMatType(description, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatIndexBase(description, CUSPARSE_INDEX_BASE_ZERO);
            cusolverSpCreateCsrqrInfo(&info);
        }

        ~cuSolverDssHandleType() {
            cusparseDestroyMatDescr(description);
            cusolverSpDestroy(handle);
            cusolverSpDestroyCsrqrInfo(info);
        }
    };
    std::shared_ptr<cuSolverDssHandleType> cusolver_dss_handle;

public:
    DSSHandle() : cusolver_dss_handle(std::make_shared<cuSolverDssHandleType>()) {}

    cusolverSpHandle_t& get_handle() { return cusolver_dss_handle->handle; }

    cusparseMatDescr_t& get_description() { return cusolver_dss_handle->description; }

    csrqrInfo_t& get_info() { return cusolver_dss_handle->info; }

    Kokkos::View<char*>& get_buffer() { return cusolver_dss_handle->buffer; }
};

}  // namespace openturbine
