#pragma once

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "netcdf_file.hpp"

namespace kynema::util {

/**
 * @brief Class for writing Kynema nodal state data to NetCDF-based output files
 *
 * This class handles the writing of nodal state data for Kynema simulations to NetCDF format.
 * It manages the output of:
 *   - Position (x, y, z, w, i, j, k)
 *   - Displacement (x, y, z, w, i, j, k)
 *   - Velocity (x, y, z, i, j, k)
 *   - Acceleration (x, y, z, i, j, k)
 *   - Force (x, y, z, i, j, k)
 *   - Deformation (x, y, z)
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
    NodeStateWriter(const std::string& file_path, bool create, size_t num_nodes);

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
    ) const;

    /**
     * @brief Write deformation data for all nodes at a timestep
     *
     * @param timestep Current timestep index
     * @param x Data for x component of deformation
     * @param y Data for y component of deformation
     * @param z Data for z component of deformation
     */
    void WriteDeformationDataAtTimestep(
        size_t timestep, const std::vector<double>& x, const std::vector<double>& y,
        const std::vector<double>& z
    ) const;

    /// @brief Get the NetCDF file object
    [[nodiscard]] const NetCDFFile& GetFile() const;

    /// @brief Get the number of nodes with state data in output file
    [[nodiscard]] size_t GetNumNodes() const;

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
    );
};

}  // namespace kynema::util
