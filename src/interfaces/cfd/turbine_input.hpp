#pragma once

#include "interfaces/cfd/floating_platform_input.hpp"

namespace openturbine::interfaces::cfd {

struct TurbineInput {
    // Floating platform
    FloatingPlatformInput floating_platform;
};

}  // namespace openturbine::interfaces::cfd
