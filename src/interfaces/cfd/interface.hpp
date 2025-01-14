/// Fluid-Structure Interaction Interface to AMR-Wind and Nalu-Wind CFD Codes

#pragma once

#include "src/interfaces/cfd/interface_input.hpp"
#include "src/interfaces/cfd/turbine.hpp"

namespace openturbine::cfd {

class Interface {
public:
    /// @brief Step forward in time
    virtual void Step() = 0;

    /// @brief Save state for correction step
    virtual void SaveState() = 0;

    /// @brief Restore state for correction step
    virtual void RestoreState() = 0;

    virtual ~Interface() = default;
    Interface() = default;
    Interface(const Interface&) = default;
    Interface& operator=(const Interface&) = default;
    Interface(Interface&&) = default;
    Interface& operator=(Interface&&) = default;
};

}  // namespace openturbine::cfd
