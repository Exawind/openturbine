#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include <stdexcept>

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
    explicit NetCDFFile(const std::string& file_path, bool create = true);

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
    ~NetCDFFile();

    //--------------------------------------------------------------------------
    // Setter/Write methods
    //--------------------------------------------------------------------------

    /**
     * @brief Adds a dimension to the NetCDF file
     *
     * This function is a wrapper around the NetCDF library's "nc_def_dim" function.
     * It creates a new dimension with the given name and length in the NetCDF file.
     */
    [[nodiscard]] int AddDimension(const std::string& name, size_t length) const;

    /**
     * @brief Adds a variable to the NetCDF file
     *
     * This function is a wrapper around the NetCDF library's "nc_def_var" function.
     * It creates a new variable with the given name and dimension IDs in the NetCDF file.
     * The NetCDF type is automatically determined from the template parameter.
     */
    template <typename T>
    [[nodiscard]] int AddVariable(const std::string& name, const std::vector<int>& dim_ids) const;

    /**
     * @brief Adds an attribute to a variable in the NetCDF file
     *
     * This function is a wrapper around the NetCDF library's "nc_put_att_text" and "nc_put_att"
     * functions. It adds an attribute (e.g. metadata) to a variable in the NetCDF file.
     */
    void AddAttribute(const std::string& var_name, const std::string& attr_name, float value)
        const;
    void AddAttribute(const std::string& var_name, const std::string& attr_name, double value)
        const;
    void AddAttribute(const std::string& var_name, const std::string& attr_name, int value)
        const;
    void AddAttribute(const std::string& var_name, const std::string& attr_name, const std::string& value)
        const;

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
    void WriteVariable(const std::string& name, const std::vector<float>& data) const;
    void WriteVariable(const std::string& name, const std::vector<double>& data) const;
    void WriteVariable(const std::string& name, const std::vector<int>& data) const;
    void WriteVariable(const std::string& name, const std::vector<std::string>& data) const;

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
    void WriteVariableAt(
        const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
        const std::vector<float>& data
    ) const;
    void WriteVariableAt(
        const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
        const std::vector<double>& data
    ) const;
    void WriteVariableAt(
        const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
        const std::vector<int>& data
    ) const;
    void WriteVariableAt(
        const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
        const std::vector<std::string>& data
    ) const;

    /// @brief Synchronizes (flushes) the NetCDF file to disk
    void Sync() const { check_netCDF_error(nc_sync(netcdf_id_), "Failed to sync NetCDF file"); }

    //--------------------------------------------------------------------------
    // Getter/Read methods
    //--------------------------------------------------------------------------

    /// @brief Returns the NetCDF file ID
    [[nodiscard]] int GetNetCDFId() const;

    /// @brief Returns the dimension ID for a given dimension name
    [[nodiscard]] int GetDimensionId(const std::string& name) const;

    /// @brief Returns the variable ID for a given variable name
    [[nodiscard]] int GetVariableId(const std::string& name) const;

    /**
     * @brief Gets the number of dimensions of a variable in the NetCDF file
     *
     * @param var_name The name of the variable
     * @return The number of dimensions
     */
    [[nodiscard]] size_t GetNumberOfDimensions(const std::string& var_name) const;

    /**
     * @brief Gets the length of a dimension in the NetCDF file
     *
     * @param dim_id The ID of the dimension
     * @return The length of the dimension
     */
    [[nodiscard]] size_t GetDimensionLength(int dim_id) const;

    /// @brief Gets the length of a dimension in the NetCDF file based on name
    [[nodiscard]] size_t GetDimensionLength(const std::string& name) const;

    /**
     * @brief Gets the shape (dimension lengths) of a variable in the NetCDF file
     *
     * @param var_name The name of the variable
     * @return Vector containing the length of each dimension of the variable
     */
    [[nodiscard]] std::vector<size_t> GetShape(const std::string& var_name) const;

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
    void ReadVariable(const std::string& name, float* data) const;
    void ReadVariable(const std::string& name, double* data) const;
    void ReadVariable(const std::string& name, int* data) const;

    /**
     * @brief Reads data from a variable at specific indices in the NetCDF file
     *
     * @tparam T The data type to read (float, double, or int)
     * @param name The name of the variable to read from
     * @param start Array specifying the starting index in each dimension
     * @param count Array specifying the number of values to read in each dimension
     * @param data Pointer to the buffer where data will be stored
     */
    void ReadVariableAt(
        const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
        float* data
    ) const;
    void ReadVariableAt(
        const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
        double* data
    ) const;
    void ReadVariableAt(
        const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
        int* data
    ) const;

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
    void ReadVariableWithStride(
        const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
        const std::vector<ptrdiff_t>& stride, float* data
    ) const;
    void ReadVariableWithStride(
        const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
        const std::vector<ptrdiff_t>& stride, double* data
    ) const;
    void ReadVariableWithStride(
        const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
        const std::vector<ptrdiff_t>& stride, int* data
    ) const;

private:
    int netcdf_id_{-1};
};

}  // namespace openturbine::util
