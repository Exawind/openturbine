#include "blade_interface.hpp"

#include "interfaces/components/blade.hpp"
#include "interfaces/components/blade_input.hpp"
#include "model/model.hpp"
#include "state/clone_state.hpp"
#include "state/copy_state_data.hpp"
#include "state/set_node_external_loads.hpp"
#include "state/state.hpp"
#include "step/step.hpp"

namespace openturbine::interfaces {

using namespace openturbine::interfaces::components;

BladeInterface::BladeInterface(SolutionInput solution_input, BladeInput blade_input)
    : model(Model(solution_input.gravity)),
      blade(blade_input, model),
      state(model.CreateState()),
      elements(model.CreateElements()),
      constraints(model.CreateConstraints()),
      parameters(
          solution_input.dynamic_solve, solution_input.max_iter, solution_input.time_step,
          solution_input.rho_inf, solution_input.absolute_error_tolerance,
          solution_input.relative_error_tolerance
      ),
      solver(CreateSolver(state, elements, constraints)),
      state_save(CloneState(state)),
      host_state(state),
      vtk_output(solution_input.vtk_output_path) {
    // Update the blade motion to match state
    UpdateNodeMotion();
}

bool BladeInterface::Step() {
    // Transfer node loads to state
    for (const auto& node : this->blade.nodes) {
        for (auto j = 0U; j < 6; ++j) {
            state.host_f(node.id, j) = node.loads[j];
        }
    }

    // Solve for state at end of step
    auto converged = openturbine::Step(
        this->parameters, this->solver, this->elements, this->state, this->constraints
    );

    // If not converged, return false
    if (!converged) {
        return false;
    }

    // Update the blade motion
    UpdateNodeMotion();

    return true;
}

void BladeInterface::SaveState() {
    // Copy current state data into saved state
    CopyStateData(this->state_save, this->state);
}

void BladeInterface::RestoreState() {
    // Restore current state from saved state
    CopyStateData(this->state, this->state_save);

    // Update the turbine motion to match restored state
    UpdateNodeMotion();
}

void BladeInterface::SetRootDisplacement(const Array_7& u) {
    if (this->blade.prescribed_root_constraint_id == kInvalidID) {
        throw("prescribed root motion was not enabled");
    }
    this->constraints.UpdateDisplacement(this->blade.prescribed_root_constraint_id, u);
}

void BladeInterface::WriteOutputVTK() {
    // this->vtk_output.WriteNodes(this->blade.nodes);
    this->vtk_output.WriteBeam(this->blade.nodes);
    this->vtk_output.IncrementFileIndex();
}

void BladeInterface::UpdateNodeMotion() {
    // Copy state motion members from device to host
    this->host_state.CopyFromState(this->state);

    // Set blade root node motion
    this->host_state.SetNodeMotion(this->blade.root_node);

    // Loop through blade nodes and set node motion
    for (auto& node : this->blade.nodes) {
        this->host_state.SetNodeMotion(node);
    }
}

}  // namespace openturbine::interfaces
