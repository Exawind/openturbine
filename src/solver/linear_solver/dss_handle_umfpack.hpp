#pragma once

#include <umfpack.h>

namespace openturbine {
template <>
class DSSHandle<DSSAlgorithm::UMFPACK> {
    struct umfpackDssHandleType {
        void* Symbolic = nullptr;
        void* Numeric = nullptr;
        double Control[UMFPACK_CONTROL];

        umfpackDssHandleType() {
            umfpack_di_defaults(Control);
            Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_BEST;
        }

        umfpackDssHandleType(umfpackDssHandleType&) = delete;
        void operator=(umfpackDssHandleType&) = delete;
        umfpackDssHandleType(umfpackDssHandleType&&) = delete;
        void operator=(umfpackDssHandleType&&) = delete;

        ~umfpackDssHandleType() {
            umfpack_di_free_symbolic(&Symbolic);
            umfpack_di_free_numeric(&Numeric);
        }
    };
    std::shared_ptr<umfpackDssHandleType> umfpack_dss_handle;

public:
    DSSHandle() : umfpack_dss_handle(std::make_shared<umfpackDssHandleType>()) {}

    void*& get_symbolic() { return umfpack_dss_handle->Symbolic; }

    void*& get_numeric() { return umfpack_dss_handle->Numeric; }

    double* get_control() { return umfpack_dss_handle->Control; }
};

}  // namespace openturbine
