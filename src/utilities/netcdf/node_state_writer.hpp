#pragma once

#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "netcdf_file.hpp"

namespace openturbine::util {

/**
 * @brief Class for writing OpenTurbine nodal state data to NetCDF-based output files
 *
 * This class handles the writing of nodal state data for OpenTurbine simulations to NetCDF format.
 * It manages the output of:
 *   - Position (x, y, z, w, i, j, k)
 *   - Displacement (x, y, z, w, i, j, k)
 *   - Velocity (x, y, z, i, j, k)
 *   - Acceleration (x, y, z, i, j, k)
 *   - Force (x, y, z, i, j, k)
 *
 * Each item is stored as a separate variable in the NetCDF file, organized by timestep
 * and node index. The file structure uses an unlimited time dimension to allow for
 * continuous writing of timesteps during simulation.
 */
class NodeStateWriter {
public:
    /**
     * @brief Constructor to create a NodeStateWriter object
     *
     * @param file_path Path to the output NetCDF file
     * @param create Whether to create a new file or open an existing one
     * @param num_nodes Number of nodes in the simulation
     */
    NodeStateWriter(const std::string& file_path, bool create, size_t num_nodes)
        : file_(file_path, create), num_nodes_(num_nodes) {
        // Define dimensions
        int time_dim = file_.AddDimension("time", NC_UNLIMITED);  // Unlimited timesteps can be added
        int node_dim = file_.AddDimension("nodes", num_nodes);
        std::vector<int> dimensions = {time_dim, node_dim};

        // Define variables for each state component
        this->DefineStateVariables("x", dimensions, true);   // Position
        this->DefineStateVariables("u", dimensions, true);   // Displacement
        this->DefineStateVariables("v", dimensions, false);  // Velocity
        this->DefineStateVariables("a", dimensions, false);  // Acceleration
        this->DefineStateVariables("f", dimensions, false);  // Force
    }

    /**
     * @brief Writes state data for a specific timestep
     *
     * @param timestep Current timestep index
     * @param component_prefix Prefix for the component
     * @param x Data for component 1
     * @param y Data for component 2
     * @param z Data for component 3
     * @param i Data for component 4
     * @param j Data for component 5
     * @param k Data for component 6
     * @param w Data for component 7 (optional, only used for position and displacement)
     */
    void WriteStateData(
        size_t timestep, const std::string& component_prefix, const std::vector<double>& x,
        const std::vector<double>& y, const std::vector<double>& z, const std::vector<double>& i,
        const std::vector<double>& j, const std::vector<double>& k,
        const std::vector<double>& w = std::vector<double>()
    ) {
        // Validate the component prefix
        static const std::array<std::string_view, 5> valid_prefixes = {"x", "u", "v", "a", "f"};
        if (std::none_of(valid_prefixes.begin(), valid_prefixes.end(), [&](const auto& prefix) {
                return prefix == component_prefix;
            })) {
            throw std::invalid_argument("Invalid component prefix: " + component_prefix);
        }

        // Validate vector sizes
        const size_t size = x.size();
        if (y.size() != size || z.size() != size || i.size() != size || j.size() != size ||
            k.size() != size) {
            throw std::invalid_argument("All vectors must have the same size");
        }

        // Write data to variables
        std::vector<size_t> start = {timestep, 0};
        std::vector<size_t> count = {1, x.size()};
        file_.WriteVariableAt(component_prefix + "_x", start, count, x);
        file_.WriteVariableAt(component_prefix + "_y", start, count, y);
        file_.WriteVariableAt(component_prefix + "_z", start, count, z);
        file_.WriteVariableAt(component_prefix + "_i", start, count, i);
        file_.WriteVariableAt(component_prefix + "_j", start, count, j);
        file_.WriteVariableAt(component_prefix + "_k", start, count, k);

        // Write w component only for position and displacement
        if (!w.empty() && (component_prefix == "x" || component_prefix == "u")) {
            file_.WriteVariableAt(component_prefix + "_w", start, count, w);
        }
    }

    const NetCDFFile& GetFile() const { return file_; }
    size_t GetNumNodes() const { return num_nodes_; }

private:
    NetCDFFile file_;
    size_t num_nodes_;

    /**
     * @brief Defines variables for a state component (position, velocity, etc.)
     *
     * @param prefix Prefix for the variable names
     * @param dims Vector of dimension IDs
     */
    void DefineStateVariables(
        const std::string& prefix, const std::vector<int>& dimensions, bool has_w
    ) {
        file_.AddVariable<double>(prefix + "_x", dimensions);
        file_.AddVariable<double>(prefix + "_y", dimensions);
        file_.AddVariable<double>(prefix + "_z", dimensions);
        file_.AddVariable<double>(prefix + "_i", dimensions);
        file_.AddVariable<double>(prefix + "_j", dimensions);
        file_.AddVariable<double>(prefix + "_k", dimensions);

        if (has_w) {
            file_.AddVariable<double>(prefix + "_w", dimensions);
        }
    }
};

}  // namespace openturbine::util
