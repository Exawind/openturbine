#pragma once

#include "interfaces/cfd/floating_platform_input.hpp"

namespace openturbine::cfd {

struct TurbineInput {
    // Floating platform
    FloatingPlatformInput floating_platform;
};

}  // namespace openturbine::cfd
