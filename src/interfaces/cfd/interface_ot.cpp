#include "src/interfaces/cfd/interface_ot.hpp"

#include "src/interfaces/cfd/floating_platform.hpp"
#include "src/interfaces/cfd/floating_platform_input.hpp"
#include "src/interfaces/cfd/mooring_line_input.hpp"
#include "src/interfaces/cfd/node_data.hpp"
#include "src/interfaces/cfd/turbine.hpp"
#include "src/interfaces/cfd/turbine_input.hpp"
#include "src/model/model.hpp"
#include "src/state/clone_state.hpp"
#include "src/state/copy_state_data.hpp"
#include "src/state/set_node_external_loads.hpp"
#include "src/state/state.hpp"
#include "src/step/step.hpp"

namespace openturbine::cfd {

//------------------------------------------------------------------------------
// Shared functions
//------------------------------------------------------------------------------

void GetNodeMotion(
    NodeData& node, const std::vector<std::vector<double>>& state_x,
    const std::vector<std::vector<double>>& state_q, const std::vector<std::vector<double>>& state_v,
    const std::vector<std::vector<double>>& state_vd
) {
    for (auto i = 0U; i < kLieGroupComponents; ++i) {
        node.position[i] = state_x[node.id][i];
        node.displacement[i] = state_q[node.id][i];
    }
    for (auto i = 0U; i < kLieAlgebraComponents; ++i) {
        node.velocity[i] = state_v[node.id][i];
        node.acceleration[i] = state_vd[node.id][i];
    }
}

//------------------------------------------------------------------------------
// Floating Platform
//------------------------------------------------------------------------------

FloatingPlatform NewFloatingPlatform(const FloatingPlatformInput& input, Model& model) {
    // Instantiate floating platform
    FloatingPlatform platform{};

    // If floating platform is not enabled, return
    if (!input.enable) {
        return platform;
    }

    // Set platform active to true
    platform.active = true;

    // Construct platform node and save ID
    platform.node.id = model.AddNode()
                           .SetPosition(input.position)
                           .SetVelocity(input.velocity)
                           .SetAcceleration(input.acceleration)
                           .Build();

    // Add element for platform mass
    platform.mass_element_id = model.AddMassElement(platform.node.id, input.mass_matrix);

    // Construct mooring lines
    std::transform(
        std::cbegin(input.mooring_lines), std::cend(input.mooring_lines),
        std::back_inserter(platform.mooring_lines),
        [&](const MooringLineInput& ml_input) {
            // Add fairlead node
            auto fairlead_node_id = model.AddNode()
                                        .SetPosition(ml_input.fairlead_position)
                                        .SetVelocity(ml_input.fairlead_velocity)
                                        .SetAcceleration(ml_input.fairlead_acceleration)
                                        .Build();

            // Add anchor node
            auto anchor_node_id = model.AddNode()
                                      .SetPosition(ml_input.anchor_position)
                                      .SetVelocity(ml_input.anchor_velocity)
                                      .SetAcceleration(ml_input.anchor_acceleration)
                                      .Build();

            // Add constraint from fairlead node to platform node
            auto rigid_constraint_id = 0U;
            //     model.AddRigidJointConstraint(platform.node.id, fairlead_node_id);

            // Add spring from fairlead to anchor
            auto spring_element_id = model.AddSpringElement(
                fairlead_node_id, anchor_node_id, ml_input.stiffness, ml_input.undeformed_length
            );

            // Add mooring line data to platform
            return MooringLine(
                fairlead_node_id, anchor_node_id, spring_element_id, rigid_constraint_id
            );
        }
    );

    return platform;
}

void SetPlatformLoads(const FloatingPlatform& platform, State& state) {
    // Return if platform is not active
    if (!platform.active) {
        return;
    }

    // Set external loads on platform node
    SetNodeExternalLoads(state, platform.node.id, platform.node.loads);
}

void GetFloatingPlatformMotion(
    FloatingPlatform& platform, const std::vector<std::vector<double>>& state_x,
    const std::vector<std::vector<double>>& state_q, const std::vector<std::vector<double>>& state_v,
    const std::vector<std::vector<double>>& state_vd
) {
    // If platform is not active, return
    if (!platform.active) {
        return;
    }

    // Populate platform node motion
    GetNodeMotion(platform.node, state_x, state_q, state_v, state_vd);

    // Loop through mooring lines
    for (auto& ml : platform.mooring_lines) {
        // Populate fairlead node motion
        GetNodeMotion(ml.fairlead_node, state_x, state_q, state_v, state_vd);

        // Populate anchor node motion
        GetNodeMotion(ml.anchor_node, state_x, state_q, state_v, state_vd);
    }
}

//------------------------------------------------------------------------------
// Turbine
//------------------------------------------------------------------------------

Turbine NewTurbine(const TurbineInput& input, Model& model) {
    return {
        NewFloatingPlatform(input.floating_platform, model),
    };
}

void GetTurbineMotion(Turbine& turbine, const State& state) {
    const auto state_x = kokkos_view_2D_to_vector(state.x);
    const auto state_q = kokkos_view_2D_to_vector(state.q);
    const auto state_v = kokkos_view_2D_to_vector(state.v);
    const auto state_vd = kokkos_view_2D_to_vector(state.vd);

    GetFloatingPlatformMotion(turbine.floating_platform, state_x, state_q, state_v, state_vd);
}

void SetTurbineLoads(const Turbine& turbine, State& state) {
    SetPlatformLoads(turbine.floating_platform, state);
}

//------------------------------------------------------------------------------
// Interface implementation
//------------------------------------------------------------------------------

InterfaceOT::InterfaceOT(const InterfaceInput& input)
    : model(),
      turbine(NewTurbine(input.turbine, model)),
      state(model.CreateState()),
      elements(model.CreateElements()),
      constraints(model.CreateConstraints()),
      parameters(true, input.max_iter, input.time_step, input.rho_inf),
      solver(CreateSolver(state, elements, constraints)),
      state_save(CloneState(state)) {
}

void InterfaceOT::Step() {
    // Transfer loads to solver
    SetTurbineLoads(this->turbine, this->state);

    // Solve for state at end of step
    auto converged = openturbine::Step(
        this->parameters, this->solver, this->elements, this->state, this->constraints
    );
    if (!converged) {
        throw std::runtime_error("Failed to converge during solver step");
    }

    // Update the turbine motion
    GetTurbineMotion(this->turbine, this->state);
}

void InterfaceOT::SaveState() {
    CopyStateData(this->state_save, this->state);
}

void InterfaceOT::RestoreState() {
    CopyStateData(this->state, this->state_save);

    // Update the turbine motion to match restored state
    GetTurbineMotion(this->turbine, this->state);
}

}  // namespace openturbine::cfd
