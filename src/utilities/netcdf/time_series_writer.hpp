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
 * @brief Class for writing time-series data to NetCDF file
 */
class TimeSeriesWriter {
public:
    /**
     * @brief Constructor to create a TimeSeriesWriter object
     *
     * @param file_path Path to the output NetCDF file
     * @param create Whether to create a new file or open an existing one
     */
    TimeSeriesWriter(const std::string& file_path, bool create = true) : file_(file_path, create) {
        try {
            time_dim_ = file_.GetDimensionId("time");
        } catch (const std::runtime_error&) {
            time_dim_ = file_.AddDimension("time", NC_UNLIMITED);
        }
    }

    /**
     * @brief Writes multiple values for a time-series variable at a specific timestep
     *
     * @tparam T The data type of the values to write
     * @param variable_name Name of the variable to write
     * @param timestep Current timestep index
     * @param values Vector of values to write at the current timestep
     */
    template <typename T>
    void WriteValues(
        const std::string& variable_name, size_t timestep, const std::vector<T>& values
    ) {
        try {
            file_.GetVariableId(variable_name);
        } catch (const std::runtime_error&) {
            int value_dim{-1};
            const std::string value_dim_name = variable_name + "_dimension";
            try {
                value_dim = file_.GetDimensionId(value_dim_name);
            } catch (const std::runtime_error&) {
                value_dim = file_.AddDimension(value_dim_name, values.size());
            }

            std::vector<int> dimensions = {time_dim_, value_dim};
            file_.AddVariable<T>(variable_name, dimensions);
        }

        std::vector<size_t> start = {timestep, 0};
        std::vector<size_t> count = {1, values.size()};
        file_.WriteVariableAt(variable_name, start, count, values);
    }

    /**
     * @brief Writes a single value for a time-series variable at a specific timestep
     *
     * @tparam T The data type of the value to write
     * @param variable_name Name of the variable to write
     * @param timestep Current timestep index
     * @param value Value to write at the current timestep
     */
    template <typename T>
    void WriteValue(const std::string& variable_name, size_t timestep, const T& value) {
        WriteValues(variable_name, timestep, std::vector<T>{value});
    }

    const NetCDFFile& GetFile() const { return file_; }

private:
    NetCDFFile file_;
    int time_dim_;
};

}  // namespace openturbine::util
