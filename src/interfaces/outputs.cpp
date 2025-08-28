#include "outputs.hpp"

#include <ranges>

#include "host_state.hpp"

namespace openturbine::interfaces {
Outputs::Outputs(const std::string& output_file, size_t num_nodes, OutputLocation location)
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

Outputs::Outputs(
    const std::string& output_file, const std::string& time_series_file, size_t num_nodes,
    OutputLocation location
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

std::unique_ptr<util::NodeStateWriter>& Outputs::GetOutputWriter() {
    return this->output_writer_;
}

std::unique_ptr<util::TimeSeriesWriter>& Outputs::GetTimeSeriesWriter() {
    return this->time_series_writer_;
}

Outputs::OutputLocation Outputs::GetLocation() const {
    return this->location_;
}

void Outputs::WriteNodeOutputsAtTimestep(const HostState<DeviceType>& host_state, size_t timestep) {
    if (!this->output_writer_) {
        return;
    }

    // Position data
    for (auto node : std::views::iota(0U, num_nodes_)) {
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
    for (auto node : std::views::iota(0U, num_nodes_)) {
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
    for (auto node : std::views::iota(0U, num_nodes_)) {
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
    for (auto node : std::views::iota(0U, num_nodes_)) {
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

void Outputs::WriteRotorTimeSeriesAtTimestep(
    size_t timestep, double azimuth_angle, double rotor_speed
) {
    if (!this->time_series_writer_) {
        return;
    }

    this->time_series_writer_->WriteValueAtTimestep("rotor_azimuth_angle", timestep, azimuth_angle);
    this->time_series_writer_->WriteValueAtTimestep("rotor_speed", timestep, rotor_speed);
}
}  // namespace openturbine::interfaces
