#pragma once

#include "interfaces/cfd/floating_platform_input.hpp"

namespace openturbine::interfaces::cfd {

/**
 * A collection of the input objects defining the CFD problem's configuration
 */
struct TurbineInput {
    // Floating platform
    FloatingPlatformInput floating_platform;
};

}  // namespace openturbine::interfaces::cfd
