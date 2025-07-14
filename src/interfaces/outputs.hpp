#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "utilities/netcdf/node_state_writer.hpp"
#include "utilities/netcdf/time_series_writer.hpp"

namespace openturbine::interfaces {

template <typename DeviceType>
struct HostState;

class Outputs {
public:
    using DeviceType =
        Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;

    /// @brief Enum for selecting where on elements to write the outputs
    enum class OutputLocation : std::uint8_t {
        kNodes = 0,  ///< Write outputs at node locations
        kQPs = 1     ///< Write outputs at quadrature points
    };

    /// @brief Constructor taking an output file and location
    Outputs(
        const std::string& output_file, size_t num_nodes,
        OutputLocation location = OutputLocation::kNodes
    );

    /// @brief Constructor taking an output file, time-series file, and location
    Outputs(
        const std::string& output_file, const std::string& time_series_file, size_t num_nodes,
        OutputLocation location = OutputLocation::kNodes
    );

    [[nodiscard]] std::unique_ptr<util::NodeStateWriter>& GetOutputWriter();

    [[nodiscard]] std::unique_ptr<util::TimeSeriesWriter>& GetTimeSeriesWriter();

    [[nodiscard]] OutputLocation GetLocation() const;

    /// @brief Write node state outputs to NetCDF file at specified timestep
    void WriteNodeOutputsAtTimestep(const HostState<DeviceType>& host_state, size_t timestep);

    /// @brief Write rotor time-series data at specified timestep
    void WriteRotorTimeSeriesAtTimestep(size_t timestep, double azimuth_angle, double rotor_speed);

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
