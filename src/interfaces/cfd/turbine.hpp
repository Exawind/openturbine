#pragma once

#include "interfaces/cfd/floating_platform.hpp"

namespace kynema::interfaces::cfd {

/**
 * @brief The top level structure defining the CFD problem
 */
struct Turbine {
    // Floating platform
    FloatingPlatform floating_platform;
};

}  // namespace kynema::interfaces::cfd
