#include "interface.hpp"

#include "floating_platform.hpp"
#include "floating_platform_input.hpp"
#include "model/model.hpp"
#include "mooring_line_input.hpp"
#include "node_data.hpp"
#include "state/clone_state.hpp"
#include "state/copy_state_data.hpp"
#include "state/read_state_from_file.hpp"
#include "state/set_node_external_loads.hpp"
#include "state/state.hpp"
#include "state/write_state_to_file.hpp"
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
      host_state_vd("host_state_vd", state.num_system_nodes),
      output_writer_(nullptr) {
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

    // Initialize NetCDF writer and write mesh connectivity if output path is specified
    if (!input.output_file.empty()) {
        // Create output directory if it doesn't exist
        std::filesystem::create_directories(input.output_file);

        // Initialize NetCDF writer
        this->output_writer_ = std::make_unique<util::NodeStateWriter>(
            input.output_file + "/cfd_interface.nc", true, state.num_system_nodes
        );

        // Write mesh connectivity to YAML file
        model.ExportMeshConnectivityToYAML(input.output_file + "/mesh_connectivity.yaml");
    }
}

void Interface::WriteRestart(const std::filesystem::path& filename) const {
    auto output = std::ofstream(filename);
    WriteStateToFile(output, state);
}

void Interface::ReadRestart(const std::filesystem::path& filename) {
    auto input = std::ifstream(filename);
    ReadStateFromFile(input, state);

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

    // Write outputs and increment timestep counter
    if (this->output_writer_) {
        WriteOutputs();
    }
    this->current_timestep_++;

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

//------------------------------------------------------------------------------
// Write outputs
//------------------------------------------------------------------------------
void Interface::WriteOutputs() const {
    const size_t num_nodes = state.num_system_nodes;
    std::vector<double> x(num_nodes);
    std::vector<double> y(num_nodes);
    std::vector<double> z(num_nodes);
    std::vector<double> i(num_nodes);
    std::vector<double> j(num_nodes);
    std::vector<double> k(num_nodes);
    std::vector<double> w(num_nodes);

    // position
    for (size_t node = 0; node < num_nodes; ++node) {
        x[node] = this->host_state_x(node, 0);
        y[node] = this->host_state_x(node, 1);
        z[node] = this->host_state_x(node, 2);
        w[node] = this->host_state_x(node, 3);
        i[node] = this->host_state_x(node, 4);
        j[node] = this->host_state_x(node, 5);
        k[node] = this->host_state_x(node, 6);
    }
    output_writer_->WriteStateDataAtTimestep(this->current_timestep_, "x", x, y, z, i, j, k, w);

    // displacement
    for (size_t node = 0; node < num_nodes; ++node) {
        x[node] = this->host_state_q(node, 0);
        y[node] = this->host_state_q(node, 1);
        z[node] = this->host_state_q(node, 2);
        w[node] = this->host_state_q(node, 3);
        i[node] = this->host_state_q(node, 4);
        j[node] = this->host_state_q(node, 5);
        k[node] = this->host_state_q(node, 6);
    }
    output_writer_->WriteStateDataAtTimestep(this->current_timestep_, "u", x, y, z, i, j, k, w);

    // velocity
    for (size_t node = 0; node < num_nodes; ++node) {
        x[node] = this->host_state_v(node, 0);
        y[node] = this->host_state_v(node, 1);
        z[node] = this->host_state_v(node, 2);
        i[node] = this->host_state_v(node, 3);
        j[node] = this->host_state_v(node, 4);
        k[node] = this->host_state_v(node, 5);
    }
    output_writer_->WriteStateDataAtTimestep(this->current_timestep_, "v", x, y, z, i, j, k);

    // acceleration
    for (size_t node = 0; node < num_nodes; ++node) {
        x[node] = this->host_state_vd(node, 0);
        y[node] = this->host_state_vd(node, 1);
        z[node] = this->host_state_vd(node, 2);
        i[node] = this->host_state_vd(node, 3);
        j[node] = this->host_state_vd(node, 4);
        k[node] = this->host_state_vd(node, 5);
    }
    output_writer_->WriteStateDataAtTimestep(this->current_timestep_, "a", x, y, z, i, j, k);
}

}  // namespace openturbine::cfd
