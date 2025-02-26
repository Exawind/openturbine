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

    // Prevent copying and moving
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
