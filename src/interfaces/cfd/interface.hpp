#pragma once

#include "interfaces/cfd/interface_input.hpp"
#include "interfaces/cfd/turbine.hpp"
#include "model/model.hpp"

namespace openturbine::cfd {

class Interface {
public:
    explicit Interface(const InterfaceInput& input);

    /// @brief Step forward in time
    [[nodiscard]] bool Step();

    /// @brief Save state for correction step
    void SaveState();

    /// @brief Restore state for correction step
    void RestoreState();

    /// @brief  OpenTurbine class used for model construction
    Model model;

    /// @brief Turbine model input/output data
    Turbine turbine;

    /// @brief  OpenTurbine class for storing system state
    State state;

    /// @brief  OpenTurbine class for model elements (beams, masses, springs)
    Elements elements;

    /// @brief  OpenTurbine class for constraints tying elements together
    Constraints constraints;

    /// @brief  OpenTurbine class containing solution parameters
    StepParameters parameters;

    /// @brief  OpenTurbine class for solving the dynamic system
    Solver solver;

    /// @brief  OpenTurbine class state class for temporarily saving state
    State state_save;

    Kokkos::View<double* [7]>::HostMirror host_state_x;
    Kokkos::View<double* [7]>::HostMirror host_state_q;
    Kokkos::View<double* [6]>::HostMirror host_state_v;
    Kokkos::View<double* [6]>::HostMirror host_state_vd;
};

}  // namespace openturbine::cfd
