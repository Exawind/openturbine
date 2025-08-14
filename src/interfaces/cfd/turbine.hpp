#pragma once

#include "interfaces/cfd/floating_platform.hpp"

namespace openturbine::interfaces::cfd {

/**
 * @brief The top level structure defining the CFD problem
 */
struct Turbine {
    // Floating platform
    FloatingPlatform floating_platform;
};

}  // namespace openturbine::interfaces::cfd
