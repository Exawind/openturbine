#pragma once

#include "interfaces/components/blade.hpp"
#include "interfaces/components/blade_input.hpp"
#include "interfaces/components/solution_input.hpp"
#include "interfaces/host_state.hpp"
#include "interfaces/vtk_output.hpp"
#include "model/model.hpp"

namespace openturbine::interfaces {

class BladeInterface {
public:
    explicit BladeInterface(
        components::SolutionInput solution_input, components::BladeInput blade_input
    );

    /// @brief Step forward in time
    [[nodiscard]] bool Step();

    /// @brief Save state for correction step
    void SaveState();

    /// @brief Restore state for correction step
    void RestoreState();

    /// @brief Set root node displacement if `prescribe_root_motion` input was true
    void SetRootDisplacement(const Array_7& u);

    void WriteOutputVTK();

private:
    /// @brief  OpenTurbine class used for model construction
    Model model;

public:
    /// @brief Blade model input/output data
    components::Blade blade;

public:
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

    /// @brief  Host local copy of node position, displacement, velocity, acceleration
    HostState host_state;

    /// @brief  VTK output manager
    VTKOutput vtk_output;

    /// @brief  Update motion data for all nodes in the interface
    void UpdateNodeMotion();
};

}  // namespace openturbine::interfaces
