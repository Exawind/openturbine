#pragma once

#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <netcdf.h>

/**
 * @brief Checks the result of a NetCDF operation and throws an exception if it fails
 */
inline void check_netCDF_error(int status, const std::string& message = "") {
    if (status != NC_NOERR) {
        throw std::runtime_error(message + ": " + nc_strerror(status));
    }
}

/**
 * @brief Class for managing NetCDF files
 */
class NetCDFFile {
public:
    /**
     * @brief Constructor to create a NetCDFFile object
     *
     * This constructor creates a new NetCDF file if the create flag is true.
     * If the create flag is false, it opens an existing NetCDF file.
     */
    explicit NetCDFFile(const std::string& file_path, bool create = true) : netcdf_id_(-1) {
        if (create) {
            check_netCDF_error(
                nc_create(file_path.c_str(), NC_CLOBBER | NC_NETCDF4, &netcdf_id_),
                "Failed to create NetCDF file"
            );
            return;
        }

        check_netCDF_error(
            nc_open(file_path.c_str(), NC_WRITE, &netcdf_id_), "Failed to open NetCDF file"
        );
    }

    /**
     * @brief Destructor to close the NetCDF file
     *
     * This function is a wrapper around the NetCDF library's nc_close function.
     * It closes the NetCDF file with the given (valid) ID.
     */
    ~NetCDFFile() {
        // Check if the NetCDF file ID is valid
        if (netcdf_id_ != -1) {
            nc_close(netcdf_id_);
        }
    }

    // Prevent copying and moving since we don't want copies made
    NetCDFFile(const NetCDFFile&) = delete;
    NetCDFFile& operator=(const NetCDFFile&) = delete;
    NetCDFFile(NetCDFFile&&) = delete;
    NetCDFFile& operator=(NetCDFFile&&) = delete;

    //--------------------------------------------------------------------------
    // Setter methods
    //--------------------------------------------------------------------------

    /**
     * @brief Adds a dimension to the NetCDF file
     *
     * This function is a wrapper around the NetCDF library's nc_def_dim function.
     * It creates a new dimension with the given name and length in the NetCDF file.
     */
    int AddDimension(const std::string& name, size_t length) {
        int dim_id;
        check_netCDF_error(
            nc_def_dim(netcdf_id_, name.c_str(), length, &dim_id),
            "Failed to create dimension " + name
        );
        return dim_id;
    }

    /**
     * @brief Adds a variable to the NetCDF file
     *
     * This function is a wrapper around the NetCDF library's nc_def_var function.
     * It creates a new variable with the given name and dimension IDs in the NetCDF file.
     * The NetCDF type is automatically determined from the template parameter.
     */
    template <typename T>
    int AddVariable(const std::string& name, const std::vector<int>& dim_ids) {
        int var_id;
        check_netCDF_error(
            nc_def_var(
                netcdf_id_, name.c_str(), GetNetCDFType<T>(), static_cast<int>(dim_ids.size()),
                dim_ids.data(), &var_id
            ),
            "Failed to create variable " + name
        );
        return var_id;
    }

    /**
     * @brief Adds an attribute to a variable in the NetCDF file
     */
    template <typename T>
    void AddAttribute(const std::string& var_name, const std::string& attr_name, const T& value) {
        int var_id = this->GetVariableId(var_name);
        if constexpr (std::is_same_v<T, std::string>) {
            check_netCDF_error(
                nc_put_att_text(
                    netcdf_id_, var_id, attr_name.c_str(), value.length(), value.c_str()
                ),
                "Failed to write attribute " + attr_name
            );
            return;
        }

        check_netCDF_error(
            nc_put_att(netcdf_id_, var_id, attr_name.c_str(), GetNetCDFType<T>(), 1, &value),
            "Failed to write attribute " + attr_name
        );
    }

    /**
     * @brief Writes data to a variable in the NetCDF file
     *
     * This function is a wrapper around the NetCDF library's nc_put_var function.
     * It writes the provided data to the variable with the given name.
     */
    template <typename T>
    void WriteVariable(const std::string& name, const std::vector<T>& data) {
        int var_id = this->GetVariableId(name);
        check_netCDF_error(
            nc_put_var(netcdf_id_, var_id, data.data()), "Failed to write variable " + name
        );
    }

    /**
     * @brief Writes data to a variable at specific indices in the NetCDF file
     *
     * @tparam T The data type of the variable
     * @param name The name of the variable to write to
     * @param start Array specifying the starting index in each dimension
     * @param count Array specifying the number of values to write in each dimension
     * @param data The vector containing the data to write
     */
    template <typename T>
    void WriteVariableAt(
        const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
        const std::vector<T>& data
    ) {
        int var_id = this->GetVariableId(name);
        check_netCDF_error(
            nc_put_vara(netcdf_id_, var_id, start.data(), count.data(), data.data()),
            "Failed to write variable " + name
        );
    }

    /**
     * @brief Synchronizes (flushes) the NetCDF file to disk
     */
    void Sync() { check_netCDF_error(nc_sync(netcdf_id_), "Failed to sync NetCDF file"); }

    //--------------------------------------------------------------------------
    // Getter methods
    //--------------------------------------------------------------------------

    /**
     * @brief Gets the NetCDF file ID
     */
    int GetNetCDFId() const { return netcdf_id_; }

    /**
     * @brief Gets the dimension ID for a given dimension name
     */
    int GetDimensionId(const std::string& name) const {
        int dim_id;
        check_netCDF_error(
            nc_inq_dimid(netcdf_id_, name.c_str(), &dim_id), "Failed to get dimension ID for " + name
        );
        return dim_id;
    }

    /**
     * @brief Gets the variable ID for a given variable name
     */
    int GetVariableId(const std::string& name) const {
        int var_id;
        check_netCDF_error(
            nc_inq_varid(netcdf_id_, name.c_str(), &var_id), "Failed to get variable ID for " + name
        );
        return var_id;
    }

private:
    int netcdf_id_;

    /**
     * @brief Helper function to convert C++ type -> NetCDF type
     *
     * @tparam T The data type to get the NetCDF type for
     * @return The NetCDF type for the given data type
     */
    template <typename T>
    static nc_type GetNetCDFType() {
        if constexpr (std::is_same_v<T, float>) {
            return NC_FLOAT;
        }
        if constexpr (std::is_same_v<T, double>) {
            return NC_DOUBLE;
        }
        if constexpr (std::is_same_v<T, int>) {
            return NC_INT;
        }
        if constexpr (std::is_same_v<T, std::string>) {
            return NC_STRING;
        }
        // Default: not supported
        throw std::runtime_error("Unsupported type");
    }
};

/**
 * @brief Class for writing state data of the nodes to NetCDF file
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
        std::vector<int> dims = {time_dim, node_dim};

        // Define variables for each state component
        this->DefineStateVariables("x", dims, true);   // Position (all 7 components)
        this->DefineStateVariables("u", dims, true);   // Displacement (all 7 components)
        this->DefineStateVariables("v", dims, false);  // Velocity (6 components)
        this->DefineStateVariables("a", dims, false);  // Acceleration (6 components)
        this->DefineStateVariables("f", dims, false);  // Force (6 components)
    }

    /**
     * @brief Writes state data for a specific timestep
     *
     * @param timestep Current timestep index
     * @param component_prefix Prefix for the component
     * @param x Data for x component
     * @param y Data for y component
     * @param z Data for z component
     * @param i Data for i component
     * @param j Data for j component
     * @param k Data for k component
     * @param w Data for w component (optional, only used for position and displacement)
     */
    void WriteStateData(
        size_t timestep, const std::string& component_prefix, const std::vector<double>& x,
        const std::vector<double>& y, const std::vector<double>& z, const std::vector<double>& i,
        const std::vector<double>& j, const std::vector<double>& k,
        const std::vector<double>& w = std::vector<double>()
    ) {
        // Validate component prefix
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
        if (!w.empty() && (component_prefix == "position" || component_prefix == "displacement")) {
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
    void DefineStateVariables(const std::string& prefix, const std::vector<int>& dims, bool has_w) {
        file_.AddVariable<double>(prefix + "_x", dims);
        file_.AddVariable<double>(prefix + "_y", dims);
        file_.AddVariable<double>(prefix + "_z", dims);
        file_.AddVariable<double>(prefix + "_i", dims);
        file_.AddVariable<double>(prefix + "_j", dims);
        file_.AddVariable<double>(prefix + "_k", dims);

        if (has_w) {
            file_.AddVariable<double>(prefix + "_w", dims);
        }
    }
};
