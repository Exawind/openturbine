#pragma once

#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <netcdf.h>

namespace openturbine::util {

/// @brief Checks the result of a NetCDF operation and throws an exception if it fails
inline void check_netCDF_error(int status, const std::string& message = "") {
    if (status != NC_NOERR) {
        throw std::runtime_error(message + ": " + nc_strerror(status));
    }
}

/// @brief Class for managing NetCDF files for writing outputs
class NetCDFFile {
public:
    /**
     * @brief Constructor to create a NetCDFFile object
     *
     * This constructor creates a new NetCDF file if the create flag is true.
     * Otherwise, it opens an existing NetCDF file.
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

    // Prevent copying and moving (since we don't want copies made of output file)
    NetCDFFile(const NetCDFFile&) = delete;
    NetCDFFile& operator=(const NetCDFFile&) = delete;
    NetCDFFile(NetCDFFile&&) = delete;
    NetCDFFile& operator=(NetCDFFile&&) = delete;

    /**
     * @brief Destructor to close the NetCDF file
     *
     * This function is a wrapper around the NetCDF library's "nc_close" function.
     * It closes the NetCDF file with the given (valid) ID.
     */
    ~NetCDFFile() {
        // Close if NetCDF file ID is valid
        if (netcdf_id_ != -1) {
            nc_close(netcdf_id_);
            netcdf_id_ = -1;
        }
    }

    //--------------------------------------------------------------------------
    // Setter/Write methods
    //--------------------------------------------------------------------------

    /**
     * @brief Adds a dimension to the NetCDF file
     *
     * This function is a wrapper around the NetCDF library's "nc_def_dim" function.
     * It creates a new dimension with the given name and length in the NetCDF file.
     */
    int AddDimension(const std::string& name, size_t length) {
        int dim_id{-1};
        check_netCDF_error(
            nc_def_dim(netcdf_id_, name.c_str(), length, &dim_id),
            "Failed to create dimension " + name
        );
        return dim_id;
    }

    /**
     * @brief Adds a variable to the NetCDF file
     *
     * This function is a wrapper around the NetCDF library's "nc_def_var" function.
     * It creates a new variable with the given name and dimension IDs in the NetCDF file.
     * The NetCDF type is automatically determined from the template parameter.
     */
    template <typename T>
    int AddVariable(const std::string& name, const std::vector<int>& dim_ids) {
        int var_id{-1};
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
     *
     * This function is a wrapper around the NetCDF library's "nc_put_att_text" and "nc_put_att"
     * functions. It adds an attribute (e.g. metadata) to a variable in the NetCDF file.
     */
    template <typename T>
    void AddAttribute(const std::string& var_name, const std::string& attr_name, const T& value) {
        int var_id = this->GetVariableId(var_name);
        // string attributes
        if constexpr (std::is_same_v<T, std::string>) {
            check_netCDF_error(
                nc_put_att_text(
                    netcdf_id_, var_id, attr_name.c_str(), value.length(), value.c_str()
                ),
                "Failed to write attribute " + attr_name
            );
            return;
        }
        // primitive/numeric attributes
        check_netCDF_error(
            nc_put_att(netcdf_id_, var_id, attr_name.c_str(), GetNetCDFType<T>(), 1, &value),
            "Failed to write attribute " + attr_name
        );
    }

    /**
     * @brief Writes data to a variable in the NetCDF file
     *
     * This function is a wrapper around the NetCDF library's "nc_put_var" and "nc_put_var_string"
     * functions. It writes the provided data to the variable with the given name.
     * Supports the following types:
     * - float (NC_FLOAT)
     * - double (NC_DOUBLE)
     * - int (NC_INT)
     * - std::string (NC_STRING)
     */
    template <typename T>
    void WriteVariable(const std::string& name, const std::vector<T>& data) {
        int var_id = this->GetVariableId(name);
        if constexpr (std::is_same_v<T, std::string>) {
            std::vector<const char*> c_strs;
            c_strs.reserve(data.size());
            for (const auto& str : data) {
                c_strs.push_back(str.c_str());
            }
            check_netCDF_error(
                nc_put_var_string(netcdf_id_, var_id, c_strs.data()),
                "Failed to write string variable " + name
            );
            return;
        }
        check_netCDF_error(
            nc_put_var(netcdf_id_, var_id, data.data()), "Failed to write variable " + name
        );
    }

    /**
     * @brief Writes data to a variable at specific indices in the NetCDF file
     *
     * This function is a wrapper around the NetCDF library's "nc_put_vara" and "nc_put_vara_string"
     * functions. It writes the provided data to the variable with the given name at the specified
     * indices. Supports the following types:
     * - float (NC_FLOAT)
     * - double (NC_DOUBLE)
     * - int (NC_INT)
     * - std::string (NC_STRING)
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
        if constexpr (std::is_same_v<T, std::string>) {
            std::vector<const char*> c_strs;
            c_strs.reserve(data.size());
            for (const auto& str : data) {
                c_strs.push_back(str.c_str());
            }
            check_netCDF_error(
                nc_put_vara_string(netcdf_id_, var_id, start.data(), count.data(), c_strs.data()),
                "Failed to write string variable " + name
            );
            return;
        }
        check_netCDF_error(
            nc_put_vara(netcdf_id_, var_id, start.data(), count.data(), data.data()),
            "Failed to write variable " + name
        );
    }

    /// @brief Synchronizes (flushes) the NetCDF file to disk
    void Sync() { check_netCDF_error(nc_sync(netcdf_id_), "Failed to sync NetCDF file"); }

    //--------------------------------------------------------------------------
    // Getter/Read methods
    //--------------------------------------------------------------------------

    /// @brief Returns the NetCDF file ID
    int GetNetCDFId() const { return netcdf_id_; }

    /// @brief Returns the dimension ID for a given dimension name
    int GetDimensionId(const std::string& name) const {
        int dim_id{-1};
        check_netCDF_error(
            nc_inq_dimid(netcdf_id_, name.c_str(), &dim_id), "Failed to get dimension ID for " + name
        );
        return dim_id;
    }

    /// @brief Returns the variable ID for a given variable name
    int GetVariableId(const std::string& name) const {
        int var_id{-1};
        check_netCDF_error(
            nc_inq_varid(netcdf_id_, name.c_str(), &var_id), "Failed to get variable ID for " + name
        );
        return var_id;
    }

    /**
     * @brief Gets the number of dimensions of a variable in the NetCDF file
     *
     * @param var_name The name of the variable
     * @return The number of dimensions
     */
    size_t GetNumberOfDimensions(const std::string& var_name) const {
        int var_id = GetVariableId(var_name);
        int num_dims{0};
        check_netCDF_error(
            nc_inq_varndims(netcdf_id_, var_id, &num_dims),
            "Failed to get number of dimensions for variable " + var_name
        );
        return static_cast<size_t>(num_dims);
    }

    /**
     * @brief Gets the length of a dimension in the NetCDF file
     *
     * @param dim_id The ID of the dimension
     * @return The length of the dimension
     */
    size_t GetDimensionLength(int dim_id) const {
        size_t length;
        check_netCDF_error(
            nc_inq_dimlen(netcdf_id_, dim_id, &length), "Failed to get dimension length"
        );
        return length;
    }

    /// @brief Gets the length of a dimension in the NetCDF file based on name
    size_t GetDimensionLength(const std::string& name) const {
        return GetDimensionLength(GetDimensionId(name));
    }

    /**
     * @brief Gets the shape (dimension lengths) of a variable in the NetCDF file
     *
     * @param var_name The name of the variable
     * @return Vector containing the length of each dimension of the variable
     */
    std::vector<size_t> GetShape(const std::string& var_name) const {
        int var_id = GetVariableId(var_name);
        size_t num_dims = GetNumberOfDimensions(var_name);

        std::vector<int> dim_ids(num_dims);
        check_netCDF_error(
            nc_inq_vardimid(netcdf_id_, var_id, dim_ids.data()),
            "Failed to get dimension IDs for variable " + var_name
        );

        std::vector<size_t> shape(num_dims);
        for (size_t i = 0; i < num_dims; ++i) {
            shape[i] = GetDimensionLength(dim_ids[i]);
        }
        return shape;
    }

    /**
     * @brief Reads data from a variable in the NetCDF file
     *
     * This function is a wrapper around the NetCDF library's type-specific get functions.
     * Supports the following types:
     * - float (NC_FLOAT)
     * - double (NC_DOUBLE)
     * - int (NC_INT)
     *
     * @param name The name of the variable to read from
     * @param data Pointer to the buffer where data will be stored
     */
    template <typename T>
    void ReadVariable(const std::string& name, T* data) const {
        int var_id = this->GetVariableId(name);
        if constexpr (std::is_same_v<T, double>) {
            check_netCDF_error(
                nc_get_var_double(netcdf_id_, var_id, data), "Failed to read double variable " + name
            );
            return;
        }
        if constexpr (std::is_same_v<T, float>) {
            check_netCDF_error(
                nc_get_var_float(netcdf_id_, var_id, data), "Failed to read float variable " + name
            );
            return;
        }
        if constexpr (std::is_same_v<T, int>) {
            check_netCDF_error(
                nc_get_var_int(netcdf_id_, var_id, data), "Failed to read int variable " + name
            );
            return;
        }

        // TODO Add support for reading std:string data

        // Default: not supported
        throw std::runtime_error("Unsupported type for reading NetCDF variable");
    }

    /**
     * @brief Reads data from a variable at specific indices in the NetCDF file
     *
     * @tparam T The data type to read (float, double, or int)
     * @param name The name of the variable to read from
     * @param start Array specifying the starting index in each dimension
     * @param count Array specifying the number of values to read in each dimension
     * @param data Pointer to the buffer where data will be stored
     */
    template <typename T>
    void ReadVariableAt(
        const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
        T* data
    ) const {
        int var_id = this->GetVariableId(name);
        if constexpr (std::is_same_v<T, double>) {
            check_netCDF_error(
                nc_get_vara_double(netcdf_id_, var_id, start.data(), count.data(), data),
                "Failed to read double variable " + name
            );
            return;
        }
        if constexpr (std::is_same_v<T, float>) {
            check_netCDF_error(
                nc_get_vara_float(netcdf_id_, var_id, start.data(), count.data(), data),
                "Failed to read float variable " + name
            );
            return;
        }
        if constexpr (std::is_same_v<T, int>) {
            check_netCDF_error(
                nc_get_vara_int(netcdf_id_, var_id, start.data(), count.data(), data),
                "Failed to read int variable " + name
            );
            return;
        }

        // TODO Add support for reading std:string data

        // Default: not supported
        throw std::runtime_error("Unsupported type for reading NetCDF variable");
    }

    /**
     * @brief Reads data from a variable with specified stride in the NetCDF file
     *
     * @tparam T The data type to read (float, double, or int)
     * @param name The name of the variable to read from
     * @param start Array specifying the starting index in each dimension
     * @param count Array specifying the number of values to read in each dimension
     * @param stride Array specifying the stride in each dimension
     * @param data Pointer to the buffer where data will be stored
     */
    template <typename T>
    void ReadVariableWithStride(
        const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
        const std::vector<ptrdiff_t>& stride, T* data
    ) const {
        int var_id = this->GetVariableId(name);
        if constexpr (std::is_same_v<T, double>) {
            check_netCDF_error(
                nc_get_vars_double(
                    netcdf_id_, var_id, start.data(), count.data(), stride.data(), data
                ),
                "Failed to read double variable with stride " + name
            );
            return;
        }
        if constexpr (std::is_same_v<T, float>) {
            check_netCDF_error(
                nc_get_vars_float(
                    netcdf_id_, var_id, start.data(), count.data(), stride.data(), data
                ),
                "Failed to read float variable with stride " + name
            );
            return;
        }
        if constexpr (std::is_same_v<T, int>) {
            check_netCDF_error(
                nc_get_vars_int(netcdf_id_, var_id, start.data(), count.data(), stride.data(), data),
                "Failed to read int variable with stride " + name
            );
            return;
        }

        // TODO Add support for reading std:string data

        // Default: not supported
        throw std::runtime_error("Unsupported type for reading NetCDF variable with stride");
    }

private:
    int netcdf_id_;

    //--------------------------------------------------------------------------
    // Helper methods
    //--------------------------------------------------------------------------

    /**
     * @brief Method to convert C++ type -> NetCDF type
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

}  // namespace openturbine::util
