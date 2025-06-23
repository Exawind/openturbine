#pragma once

#include <memory>
#include <string>
#include <vector>

#include "interfaces/host_state.hpp"
#include "utilities/netcdf/node_state_writer.hpp"
#include "utilities/netcdf/time_series_writer.hpp"

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
          location_(location) {
        this->x_data_.resize(num_nodes_);
        this->y_data_.resize(num_nodes_);
        this->z_data_.resize(num_nodes_);
        this->i_data_.resize(num_nodes_);
        this->j_data_.resize(num_nodes_);
        this->k_data_.resize(num_nodes_);
        this->w_data_.resize(num_nodes_);
    }

    /// @brief Constructor taking an output file, time-series file, and location
    Outputs(
        const std::string& output_file, const std::string& time_series_file, size_t num_nodes,
        OutputLocation location = OutputLocation::kNodes
    )
        : output_writer_(std::make_unique<util::NodeStateWriter>(output_file, true, num_nodes)),
          time_series_writer_(std::make_unique<util::TimeSeriesWriter>(time_series_file, true)),
          num_nodes_(num_nodes),
          location_(location) {
        this->x_data_.resize(num_nodes_);
        this->y_data_.resize(num_nodes_);
        this->z_data_.resize(num_nodes_);
        this->i_data_.resize(num_nodes_);
        this->j_data_.resize(num_nodes_);
        this->k_data_.resize(num_nodes_);
        this->w_data_.resize(num_nodes_);
    }

    [[nodiscard]] std::unique_ptr<util::NodeStateWriter>& GetOutputWriter() {
        return this->output_writer_;
    }

    [[nodiscard]] std::unique_ptr<util::TimeSeriesWriter>& GetTimeSeriesWriter() {
        return this->time_series_writer_;
    }

    [[nodiscard]] OutputLocation GetLocation() const { return this->location_; }

    /// @brief Write node state outputs to NetCDF file at specified timestep
    template <typename DeviceType>
    void WriteNodeOutputsAtTimestep(const HostState<DeviceType>& host_state, size_t timestep) {
        if (!this->output_writer_) {
            return;
        }

        // Position data
        for (size_t node = 0; node < num_nodes_; ++node) {
            this->x_data_[node] = host_state.x(node, 0);
            this->y_data_[node] = host_state.x(node, 1);
            this->z_data_[node] = host_state.x(node, 2);
            this->w_data_[node] = host_state.x(node, 3);
            this->i_data_[node] = host_state.x(node, 4);
            this->j_data_[node] = host_state.x(node, 5);
            this->k_data_[node] = host_state.x(node, 6);
        }
        this->output_writer_->WriteStateDataAtTimestep(
            timestep, "x", this->x_data_, this->y_data_, this->z_data_, this->i_data_, this->j_data_,
            this->k_data_, this->w_data_
        );

        // Displacement data
        for (size_t node = 0; node < num_nodes_; ++node) {
            this->x_data_[node] = host_state.q(node, 0);
            this->y_data_[node] = host_state.q(node, 1);
            this->z_data_[node] = host_state.q(node, 2);
            this->w_data_[node] = host_state.q(node, 3);
            this->i_data_[node] = host_state.q(node, 4);
            this->j_data_[node] = host_state.q(node, 5);
            this->k_data_[node] = host_state.q(node, 6);
        }
        this->output_writer_->WriteStateDataAtTimestep(
            timestep, "u", this->x_data_, this->y_data_, this->z_data_, this->i_data_, this->j_data_,
            this->k_data_, this->w_data_
        );

        // Velocity data
        for (size_t node = 0; node < num_nodes_; ++node) {
            this->x_data_[node] = host_state.v(node, 0);
            this->y_data_[node] = host_state.v(node, 1);
            this->z_data_[node] = host_state.v(node, 2);
            this->i_data_[node] = host_state.v(node, 3);
            this->j_data_[node] = host_state.v(node, 4);
            this->k_data_[node] = host_state.v(node, 5);
        }
        this->output_writer_->WriteStateDataAtTimestep(
            timestep, "v", this->x_data_, this->y_data_, this->z_data_, this->i_data_, this->j_data_,
            this->k_data_
        );

        // Acceleration data
        for (size_t node = 0; node < num_nodes_; ++node) {
            this->x_data_[node] = host_state.vd(node, 0);
            this->y_data_[node] = host_state.vd(node, 1);
            this->z_data_[node] = host_state.vd(node, 2);
            this->i_data_[node] = host_state.vd(node, 3);
            this->j_data_[node] = host_state.vd(node, 4);
            this->k_data_[node] = host_state.vd(node, 5);
        }
        this->output_writer_->WriteStateDataAtTimestep(
            timestep, "a", this->x_data_, this->y_data_, this->z_data_, this->i_data_, this->j_data_,
            this->k_data_
        );
    }

    /// @brief Write rotor time-series data at specified timestep
    void WriteRotorTimeSeriesAtTimestep(size_t timestep, double azimuth_angle, double rotor_speed) {
        if (!this->time_series_writer_) {
            return;
        }

        this->time_series_writer_->WriteValueAtTimestep(
            "rotor_azimuth_angle", timestep, azimuth_angle
        );
        this->time_series_writer_->WriteValueAtTimestep("rotor_speed", timestep, rotor_speed);
    }

private:
    std::unique_ptr<util::NodeStateWriter> output_writer_;        ///< Output writer
    std::unique_ptr<util::TimeSeriesWriter> time_series_writer_;  ///< Time series writer
    size_t num_nodes_;         ///< Number of nodes to be written in the output file
    OutputLocation location_;  ///< Output writing location in element

    // Pre-allocated vectors for storing output data
    std::vector<double> x_data_;
    std::vector<double> y_data_;
    std::vector<double> z_data_;
    std::vector<double> i_data_;
    std::vector<double> j_data_;
    std::vector<double> k_data_;
    std::vector<double> w_data_;
};

}  // namespace openturbine::interfaces
