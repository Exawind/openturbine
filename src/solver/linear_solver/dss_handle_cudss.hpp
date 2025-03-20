#pragma once

#include <cudss.h>

namespace openturbine {
template <>
class DSSHandle<DSSAlgorithm::CUDSS> {
    struct cudssDssHandleType {
        cudssHandle_t handle;
        cudssConfig_t solverConfig;
        cudssData_t solverData;

        cudssDssHandleType() {
            cudssCreate(&handle);
            cudssConfigCreate(&solverConfig);
            cudssDataCreate(handle, &solverData);
        }

        ~cudssDssHandleType() {
            cudssDataDestroy(handle, solverData);
            cudssConfigDestroy(solverConfig);
            cudssDestroy(handle);
        }
    };
    std::shared_ptr<cudssDssHandleType> cudss_dss_handle;

public:
    DSSHandle() : cudss_dss_handle(std::make_shared<cudssDssHandleType>()) {}

    cudssHandle_t& get_handle() { return cudss_dss_handle->handle; }

    cudssConfig_t& get_config() { return cudss_dss_handle->solverConfig; }

    cudssData_t& get_data() { return cudss_dss_handle->solverData; }
};

}  // namespace openturbine
