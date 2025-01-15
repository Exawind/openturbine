#include "src/interfaces/cfd/interface.hpp"

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
    NodeData& node, const Kokkos::View<double* [7]>::HostMirror::const_type& host_state_x,
    const Kokkos::View<double* [7]>::HostMirror::const_type& host_state_q,
    const Kokkos::View<double* [6]>::HostMirror::const_type& host_state_v,
    const Kokkos::View<double* [6]>::HostMirror::const_type& host_state_vd
) {
    for (auto i = 0U; i < kLieGroupComponents; ++i) {
        node.position[i] = host_state_x(node.id, i);
        node.displacement[i] = host_state_q(node.id, i);
    }
    for (auto i = 0U; i < kLieAlgebraComponents; ++i) {
        node.velocity[i] = host_state_v(node.id, i);
        node.acceleration[i] = host_state_vd(node.id, i);
    }
}

//------------------------------------------------------------------------------
// Floating Platform
//------------------------------------------------------------------------------

FloatingPlatform CreateFloatingPlatform(const FloatingPlatformInput& input, Model& model) {
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
    FloatingPlatform& platform,
    const Kokkos::View<double* [7]>::HostMirror::const_type& host_state_x,
    const Kokkos::View<double* [7]>::HostMirror::const_type& host_state_q,
    const Kokkos::View<double* [6]>::HostMirror::const_type& host_state_v,
    const Kokkos::View<double* [6]>::HostMirror::const_type& host_state_vd
) {
    // If platform is not active, return
    if (!platform.active) {
        return;
    }

    // Populate platform node motion
    GetNodeMotion(platform.node, host_state_x, host_state_q, host_state_v, host_state_vd);

    // Loop through mooring lines
    for (auto& ml : platform.mooring_lines) {
        // Populate fairlead node motion
        GetNodeMotion(ml.fairlead_node, host_state_x, host_state_q, host_state_v, host_state_vd);

        // Populate anchor node motion
        GetNodeMotion(ml.anchor_node, host_state_x, host_state_q, host_state_v, host_state_vd);
    }
}

//------------------------------------------------------------------------------
// Turbine
//------------------------------------------------------------------------------

Turbine CreateTurbine(const TurbineInput& input, Model& model) {
    return {
        CreateFloatingPlatform(input.floating_platform, model),
    };
}

void SetTurbineLoads(const Turbine& turbine, State& state) {
    SetPlatformLoads(turbine.floating_platform, state);
}

void GetTurbineMotion(
    Turbine& turbine, const Kokkos::View<double* [7]>::HostMirror::const_type& host_state_x,
    const Kokkos::View<double* [7]>::HostMirror::const_type& host_state_q,
    const Kokkos::View<double* [6]>::HostMirror::const_type& host_state_v,
    const Kokkos::View<double* [6]>::HostMirror::const_type& host_state_vd
) {
    GetFloatingPlatformMotion(
        turbine.floating_platform, host_state_x, host_state_q, host_state_v, host_state_vd
    );
}

//------------------------------------------------------------------------------
// Interface implementation
//------------------------------------------------------------------------------

Interface::Interface(const InterfaceInput& input)
    : model(input.gravity),
      turbine(CreateTurbine(input.turbine, model)),
      state(model.CreateState()),
      elements(model.CreateElements()),
      constraints(model.CreateConstraints()),
      parameters(true, input.max_iter, input.time_step, input.rho_inf),
      solver(CreateSolver(state, elements, constraints)),
      state_save(CloneState(state)),
      host_state_x("host_state_x", state.num_system_nodes),
      host_state_q("host_state_q", state.num_system_nodes),
      host_state_v("host_state_v", state.num_system_nodes),
      host_state_vd("host_state_vd", state.num_system_nodes) {
    // Copy state motion members from device to host
    Kokkos::deep_copy(this->host_state_x, this->state.x);
    Kokkos::deep_copy(this->host_state_q, this->state.q);
    Kokkos::deep_copy(this->host_state_v, this->state.v);
    Kokkos::deep_copy(this->host_state_vd, this->state.vd);

    // Update the turbine motion to match restored state
    GetTurbineMotion(
        this->turbine, this->host_state_x, this->host_state_q, this->host_state_v,
        this->host_state_vd
    );
}

void Interface::Step() {
    // Transfer loads to solver
    SetTurbineLoads(this->turbine, this->state);

    // Solve for state at end of step
    auto converged = openturbine::Step(
        this->parameters, this->solver, this->elements, this->state, this->constraints
    );
    if (!converged) {
        throw std::runtime_error("Failed to converge during solver step");
    }

    // Copy state motion members from device to host
    Kokkos::deep_copy(this->host_state_x, this->state.x);
    Kokkos::deep_copy(this->host_state_q, this->state.q);
    Kokkos::deep_copy(this->host_state_v, this->state.v);
    Kokkos::deep_copy(this->host_state_vd, this->state.vd);

    // Update the turbine motion
    GetTurbineMotion(
        this->turbine, this->host_state_x, this->host_state_q, this->host_state_v,
        this->host_state_vd
    );
}

void Interface::SaveState() {
    CopyStateData(this->state_save, this->state);
}

void Interface::RestoreState() {
    CopyStateData(this->state, this->state_save);

    // Copy state motion members from device to host
    Kokkos::deep_copy(this->host_state_x, this->state.x);
    Kokkos::deep_copy(this->host_state_q, this->state.q);
    Kokkos::deep_copy(this->host_state_v, this->state.v);
    Kokkos::deep_copy(this->host_state_vd, this->state.vd);

    // Update the turbine motion to match restored state
    GetTurbineMotion(
        this->turbine, this->host_state_x, this->host_state_q, this->host_state_v,
        this->host_state_vd
    );
}

}  // namespace openturbine::cfd
