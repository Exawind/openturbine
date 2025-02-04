#include "interface.hpp"

#include "floating_platform.hpp"
#include "floating_platform_input.hpp"
#include "model/model.hpp"
#include "mooring_line_input.hpp"
#include "node_data.hpp"
#include "state/clone_state.hpp"
#include "state/copy_state_data.hpp"
#include "state/set_node_external_loads.hpp"
#include "state/state.hpp"
#include "step/step.hpp"
#include "turbine.hpp"
#include "turbine_input.hpp"

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
    // If floating platform is not enabled, return
    if (!input.enable) {
        return {
            false,         // active
            NodeData(0U),  // platform node
            0U,            // mass element ID
            {},
        };
    }

    // Construct platform node and save ID
    const auto platform_node_id = model.AddNode()
                                      .SetPosition(input.position)
                                      .SetVelocity(input.velocity)
                                      .SetAcceleration(input.acceleration)
                                      .Build();

    // Add element for platform mass
    const auto mass_element_id = model.AddMassElement(platform_node_id, input.mass_matrix);

    // Instantiate platform
    FloatingPlatform platform{
        true,  // enable platform
        NodeData(platform_node_id),
        mass_element_id,
        {},
    };

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

            // Add fixed constraint to anchor node
            auto fixed_constraint_id = model.AddFixedBC3DOFs(anchor_node_id);

            // Add rigid constraint from fairlead node to platform node
            auto rigid_constraint_id =
                model.AddRigidJoint6DOFsTo3DOFs({platform.node.id, fairlead_node_id});

            // Add spring from fairlead to anchor
            auto spring_element_id = model.AddSpringElement(
                fairlead_node_id, anchor_node_id, ml_input.stiffness, ml_input.undeformed_length
            );

            // Add mooring line data to platform
            return MooringLine{
                NodeData(fairlead_node_id), NodeData(anchor_node_id), fixed_constraint_id,
                rigid_constraint_id,        spring_element_id,
            };
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

void Interface::WriteRestart(const std::filesystem::path& filename) const {
    auto output = std::ofstream(filename);
    auto num_system_nodes = state.num_system_nodes;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    output.write(reinterpret_cast<char*>(&num_system_nodes), sizeof(size_t));

    auto mirror_7 = Kokkos::View<double* [7]>::HostMirror("mirror_7", num_system_nodes);
    auto out_7 = Kokkos::View<double* [7], Kokkos::HostSpace>("out_7", num_system_nodes);

    auto mirror_6 = Kokkos::View<double* [6]>::HostMirror("mirror_6", num_system_nodes);
    auto out_6 = Kokkos::View<double* [6], Kokkos::HostSpace>("out_6", num_system_nodes);

    auto write_7 = [&](const Kokkos::View<double* [7]>& data) {
        Kokkos::deep_copy(mirror_7, data);
        Kokkos::deep_copy(out_7, mirror_7);

        const auto stream_size = static_cast<long>(7U * num_system_nodes * sizeof(double));
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        output.write(reinterpret_cast<char*>(out_7.data()), stream_size);
    };

    auto write_6 = [&](const Kokkos::View<double* [6]>& data) {
        Kokkos::deep_copy(mirror_6, data);
        Kokkos::deep_copy(out_6, mirror_6);

        const auto stream_size = static_cast<long>(6U * num_system_nodes * sizeof(double));
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        output.write(reinterpret_cast<char*>(out_6.data()), stream_size);
    };

    write_7(state.x0);
    write_7(state.x);
    write_6(state.q_delta);
    write_7(state.q_prev);
    write_7(state.q);
    write_6(state.v);
    write_6(state.vd);
    write_6(state.a);
    write_6(state.f);
}

void Interface::ReadRestart(const std::filesystem::path& filename) {
    auto input = std::ifstream(filename);
    auto num_system_nodes = size_t{};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    input.read(reinterpret_cast<char*>(&num_system_nodes), sizeof(size_t));

    if (num_system_nodes != state.num_system_nodes) {
        throw std::length_error("Number of system nodes in file is not the same as in model");
    }

    auto mirror_7 = Kokkos::View<double* [7]>::HostMirror("mirror_7", num_system_nodes);
    auto out_7 = Kokkos::View<double* [7], Kokkos::HostSpace>("out_7", num_system_nodes);

    auto mirror_6 = Kokkos::View<double* [6]>::HostMirror("mirror_6", num_system_nodes);
    auto out_6 = Kokkos::View<double* [6], Kokkos::HostSpace>("out_6", num_system_nodes);

    auto read_7 = [&](const Kokkos::View<double* [7]>& data) {
        const auto stream_size = static_cast<long>(7U * num_system_nodes * sizeof(double));
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        input.read(reinterpret_cast<char*>(out_7.data()), stream_size);

        Kokkos::deep_copy(mirror_7, out_7);
        Kokkos::deep_copy(data, mirror_7);
    };

    auto read_6 = [&](const Kokkos::View<double* [6]>& data) {
        const auto stream_size = static_cast<long>(6U * num_system_nodes * sizeof(double));
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        input.read(reinterpret_cast<char*>(out_6.data()), stream_size);

        Kokkos::deep_copy(mirror_6, out_6);
        Kokkos::deep_copy(data, mirror_6);
    };

    read_7(state.x0);
    read_7(state.x);
    read_6(state.q_delta);
    read_7(state.q_prev);
    read_7(state.q);
    read_6(state.v);
    read_6(state.vd);
    read_6(state.a);
    read_6(state.f);

    Kokkos::deep_copy(this->host_state_x, this->state.x);
    Kokkos::deep_copy(this->host_state_q, this->state.q);
    Kokkos::deep_copy(this->host_state_v, this->state.v);
    Kokkos::deep_copy(this->host_state_vd, this->state.vd);

    GetTurbineMotion(
        this->turbine, this->host_state_x, this->host_state_q, this->host_state_v,
        this->host_state_vd
    );
}

bool Interface::Step() {
    // Transfer loads to solver
    SetTurbineLoads(this->turbine, this->state);

    // Solve for state at end of step
    auto converged = openturbine::Step(
        this->parameters, this->solver, this->elements, this->state, this->constraints
    );
    if (!converged) {
        return false;
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

    return true;
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
