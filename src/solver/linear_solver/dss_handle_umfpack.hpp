#pragma once

#include <umfpack.h>

namespace openturbine {
template <>
class DSSHandle<DSSAlgorithm::UMFPACK> {
    struct umfpackDssHandleType {
        void* Symbolic;
        void* Numeric;

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
};

}  // namespace openturbine
