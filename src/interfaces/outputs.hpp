#pragma once

#include <memory>
#include <string>
#include <vector>

#include "interfaces/host_state.hpp"
#include "utilities/netcdf/node_state_writer.hpp"

namespace openturbine::interfaces {

class Outputs {
public:
    /// @brief Enum for selecting where on elements to write the outputs
    enum class OutputLocation : std::uint8_t {
        kNodes = 0,  ///< Write outputs at node locations
        kQPs = 1     ///< Write outputs at quadrature points
    };

    /// @brief Constructor taking an output file and location
    Outputs(
        const std::string& output_file, size_t num_nodes,
        OutputLocation location = OutputLocation::kNodes
    )
        : output_writer_(std::make_unique<util::NodeStateWriter>(output_file, true, num_nodes)),
          num_nodes_(num_nodes),
          location_(location) {}

    [[nodiscard]] std::unique_ptr<util::NodeStateWriter>& GetOutputWriter() {
        return this->output_writer_;
    }

    [[nodiscard]] OutputLocation GetLocation() const { return this->location_; }

    /// @brief Write node state outputs to NetCDF file at specified timestep
    void WriteNodeOutputsAtTimestep(const HostState& host_state, size_t timestep) const {
        if (!this->output_writer_) {
            return;
        }

        std::vector<double> x(num_nodes_);
        std::vector<double> y(num_nodes_);
        std::vector<double> z(num_nodes_);
        std::vector<double> i(num_nodes_);
        std::vector<double> j(num_nodes_);
        std::vector<double> k(num_nodes_);
        std::vector<double> w(num_nodes_);

        // Position data
        for (size_t node = 0; node < num_nodes_; ++node) {
            x[node] = host_state.x(node, 0);
            y[node] = host_state.x(node, 1);
            z[node] = host_state.x(node, 2);
            w[node] = host_state.x(node, 3);
            i[node] = host_state.x(node, 4);
            j[node] = host_state.x(node, 5);
            k[node] = host_state.x(node, 6);
        }
        this->output_writer_->WriteStateDataAtTimestep(timestep, "x", x, y, z, i, j, k, w);

        // Displacement data
        for (size_t node = 0; node < num_nodes_; ++node) {
            x[node] = host_state.q(node, 0);
            y[node] = host_state.q(node, 1);
            z[node] = host_state.q(node, 2);
            w[node] = host_state.q(node, 3);
            i[node] = host_state.q(node, 4);
            j[node] = host_state.q(node, 5);
            k[node] = host_state.q(node, 6);
        }
        this->output_writer_->WriteStateDataAtTimestep(timestep, "u", x, y, z, i, j, k, w);

        // Velocity data
        for (size_t node = 0; node < num_nodes_; ++node) {
            x[node] = host_state.v(node, 0);
            y[node] = host_state.v(node, 1);
            z[node] = host_state.v(node, 2);
            i[node] = host_state.v(node, 3);
            j[node] = host_state.v(node, 4);
            k[node] = host_state.v(node, 5);
        }
        this->output_writer_->WriteStateDataAtTimestep(timestep, "v", x, y, z, i, j, k);

        // Acceleration data
        for (size_t node = 0; node < num_nodes_; ++node) {
            x[node] = host_state.vd(node, 0);
            y[node] = host_state.vd(node, 1);
            z[node] = host_state.vd(node, 2);
            i[node] = host_state.vd(node, 3);
            j[node] = host_state.vd(node, 4);
            k[node] = host_state.vd(node, 5);
        }
        this->output_writer_->WriteStateDataAtTimestep(timestep, "a", x, y, z, i, j, k);
    }

private:
    std::unique_ptr<util::NodeStateWriter> output_writer_;  ///< Output writer
    size_t num_nodes_;         ///< Number of nodes to be written in the output file
    OutputLocation location_;  ///< Output writing location in element
};

}  // namespace openturbine::interfaces
