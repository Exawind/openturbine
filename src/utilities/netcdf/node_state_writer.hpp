#pragma once

#include <algorithm>
#include <array>
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
        // Define dimensions for time and nodes
        // NOTE: We are assuming these dimensions were not added previously
        const int time_dim =
            file_.AddDimension("time", NC_UNLIMITED);  // Unlimited timesteps can be added
        const int node_dim = file_.AddDimension("nodes", num_nodes);
        const std::vector<int> dimensions = {time_dim, node_dim};

        // Define variables for each state component
        this->DefineStateVariables("x", dimensions, true);   // Position (x, y, z, w, i, j, k)
        this->DefineStateVariables("u", dimensions, true);   // Displacement (x, y, z, w, i, j, k)
        this->DefineStateVariables("v", dimensions, false);  // Velocity (x, y, z, i, j, k)
        this->DefineStateVariables("a", dimensions, false);  // Acceleration (x, y, z, i, j, k)
        this->DefineStateVariables("f", dimensions, false);  // Force (x, y, z, i, j, k)
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
    void WriteStateDataAtTimestep(
        size_t timestep, const std::string& component_prefix, const std::vector<double>& x,
        const std::vector<double>& y, const std::vector<double>& z, const std::vector<double>& i,
        const std::vector<double>& j, const std::vector<double>& k,
        const std::vector<double>& w = std::vector<double>()
    ) const {
        // Validate the component prefix - must be one of the valid prefixes:
        // "x" -> position
        // "u" -> displacement
        // "v" -> velocity
        // "a" -> acceleration
        // "f" -> force
        static const std::array<std::string_view, 5> valid_prefixes = {"x", "u", "v", "a", "f"};
        if (std::none_of(valid_prefixes.begin(), valid_prefixes.end(), [&](const auto& prefix) {
                return prefix == component_prefix;
            })) {
            throw std::invalid_argument("Invalid component prefix: " + component_prefix);
        }

        // Validate vector sizes - must be the same for all components
        const size_t size = x.size();
        if (y.size() != size || z.size() != size || i.size() != size || j.size() != size ||
            k.size() != size) {
            throw std::invalid_argument("All vectors must have the same size");
        }

        // Write data to variables
        const std::vector<size_t> start = {timestep, 0};  // start at the current timestep and node 0
        const std::vector<size_t> count = {
            1, x.size()
        };  // write one timestep worth of data for all nodes
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

    /// @brief Get the NetCDF file object
    [[nodiscard]] const NetCDFFile& GetFile() const { return file_; }

    /// @brief Get the number of nodes with state data in output file
    [[nodiscard]] size_t GetNumNodes() const { return num_nodes_; }

private:
    NetCDFFile file_;
    size_t num_nodes_;

    /**
     * @brief Defines variables for a state component (position, velocity, etc.)
     *
     * @param prefix Prefix for the variable names
     * @param dimensions Vector of dimension IDs
     * @param has_w Whether the component has a w component
     */
    void DefineStateVariables(
        const std::string& prefix, const std::vector<int>& dimensions, bool has_w
    ) {
        (void)file_.AddVariable<double>(prefix + "_x", dimensions);
        (void)file_.AddVariable<double>(prefix + "_y", dimensions);
        (void)file_.AddVariable<double>(prefix + "_z", dimensions);
        (void)file_.AddVariable<double>(prefix + "_i", dimensions);
        (void)file_.AddVariable<double>(prefix + "_j", dimensions);
        (void)file_.AddVariable<double>(prefix + "_k", dimensions);

        if (has_w) {
            (void)file_.AddVariable<double>(prefix + "_w", dimensions);
        }
    }
};

}  // namespace openturbine::util
