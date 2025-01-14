#pragma once

#include "src/interfaces/cfd/floating_platform.hpp"

namespace openturbine::cfd {

struct Turbine {
    // Floating platform
    FloatingPlatform floating_platform;
};

}  // namespace openturbine::cfd