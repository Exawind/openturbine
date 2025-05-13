#pragma once

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "elements/elements.hpp"
#include "state/state.hpp"
#include "utilities/netcdf/node_state_writer.hpp"

namespace openturbine {

class Outputs {
public:
    /// @brief Enum for selecting where on elements to write the outputs
    enum class OutputLocation {
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
    void WriteNodeOutputsAtTimestep(State& state, size_t timestep) const {
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

        auto host_q = Kokkos::create_mirror_view(state.q);
        auto host_v = Kokkos::create_mirror_view(state.v);
        auto host_vd = Kokkos::create_mirror_view(state.vd);
        auto host_x = Kokkos::create_mirror_view(state.x);

        Kokkos::deep_copy(host_q, state.q);
        Kokkos::deep_copy(host_v, state.v);
        Kokkos::deep_copy(host_vd, state.vd);
        Kokkos::deep_copy(host_x, state.x);

        // Position data
        for (size_t node = 0; node < num_nodes_; ++node) {
            x[node] = host_x(node, 0);
            y[node] = host_x(node, 1);
            z[node] = host_x(node, 2);
            w[node] = host_x(node, 3);
            i[node] = host_x(node, 4);
            j[node] = host_x(node, 5);
            k[node] = host_x(node, 6);
        }
        this->output_writer_->WriteStateDataAtTimestep(timestep, "x", x, y, z, i, j, k, w);

        // Displacement data
        for (size_t node = 0; node < num_nodes_; ++node) {
            x[node] = host_q(node, 0);
            y[node] = host_q(node, 1);
            z[node] = host_q(node, 2);
            w[node] = host_q(node, 3);
            i[node] = host_q(node, 4);
            j[node] = host_q(node, 5);
            k[node] = host_q(node, 6);
        }
        this->output_writer_->WriteStateDataAtTimestep(timestep, "u", x, y, z, i, j, k, w);

        // Velocity data
        for (size_t node = 0; node < num_nodes_; ++node) {
            x[node] = host_v(node, 0);
            y[node] = host_v(node, 1);
            z[node] = host_v(node, 2);
            i[node] = host_v(node, 3);
            j[node] = host_v(node, 4);
            k[node] = host_v(node, 5);
        }
        this->output_writer_->WriteStateDataAtTimestep(timestep, "v", x, y, z, i, j, k);

        // Acceleration data
        for (size_t node = 0; node < num_nodes_; ++node) {
            x[node] = host_vd(node, 0);
            y[node] = host_vd(node, 1);
            z[node] = host_vd(node, 2);
            i[node] = host_vd(node, 3);
            j[node] = host_vd(node, 4);
            k[node] = host_vd(node, 5);
        }
        this->output_writer_->WriteStateDataAtTimestep(timestep, "a", x, y, z, i, j, k);
    }

private:
    std::unique_ptr<util::NodeStateWriter> output_writer_;  ///< Output writer
    size_t num_nodes_;         ///< Number of nodes to be written in the output file
    OutputLocation location_;  ///< Output writing location in element
};

}  // namespace openturbine
