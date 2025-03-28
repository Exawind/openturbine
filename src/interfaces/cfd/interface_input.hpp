#pragma once

#include "interfaces/cfd/turbine_input.hpp"
#include "types.hpp"

namespace openturbine::cfd {

struct InterfaceInput {
    /// @brief Array of gravity components (XYZ)
    std::array<double, 3> gravity{0., 0., 0.};

    /// @brief Solver time step
    double time_step{0.01};

    /// @brief Solver numerical damping factor (0 = maximum damping)
    double rho_inf{0.};

    /// @brief Maximum number of convergence iterations
    size_t max_iter{5U};

    /// @brief Turbine input data
    TurbineInput turbine;

    /// @brief Output file path for NetCDF results (empty = no outputs will be written)
    std::string output_file;
};

}  // namespace openturbine::cfd
