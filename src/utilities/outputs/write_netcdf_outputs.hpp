#pragma once

#include <chrono>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <netcdf.h>

class NetCDFWriter {
public:
    /**
     * @brief Constructor for NetCDFWriter
     * @param file_path Path to the NetCDF file
     * @param num_nodes Number of nodes
     * @param dt Time step
     * @param total_time Total simulation time
     */
    NetCDFWriter(
        const std::string& file_path = "test.nc", int num_nodes = 100, double dt = 0.01,
        double total_time = 5000.
    )
        : file_path(file_path),
          num_nodes(num_nodes),
          total_iterations(static_cast<int>(total_time / dt)),
          gen(rd()),
          dis(0., 1.) {
        // Remove existing file
        std::filesystem::remove(file_path);

        // Create new file
        int status;
        if ((status = nc_create(file_path.c_str(), NC_NETCDF4, &file_name)) != NC_NOERR) {
            throw std::runtime_error(nc_strerror(status));
        }

        // Define dimensions
        int time_dimension_id, node_dimension_id;
        if ((status = nc_def_dim(
                 file_name, "time", static_cast<size_t>(total_iterations), &time_dimension_id
             )) != NC_NOERR ||
            (status =
                 nc_def_dim(file_name, "nodes", static_cast<size_t>(num_nodes), &node_dimension_id)
            ) != NC_NOERR) {
            throw std::runtime_error(nc_strerror(status));
        }

        // Define dimension arrays for variables
        int dimensions[2] = {time_dimension_id, node_dimension_id};

        // Define variables
        DefineVariables("x", dimensions, x_var_id);
        DefineVariables("u", dimensions, u_var_id);
        DefineVariables("v", dimensions, v_var_id);
        DefineVariables("a", dimensions, a_var_id);
        DefineVariables("f", dimensions, f_var_id);

        // End define mode
        if ((status = nc_enddef(file_name)) != NC_NOERR) {
            throw std::runtime_error(nc_strerror(status));
        }
    }

    /**
     * @brief Insert random data into the file
     */
    void InsertRandomData() {
        auto commit_frequency = 1;

        std::vector<double> data(static_cast<size_t>(num_nodes));
        size_t start[2] = {0, 0};
        size_t count[2] = {1, static_cast<size_t>(num_nodes)};

        for (int t = 0; t < total_iterations; t++) {
            start[0] = static_cast<size_t>(t);

            // Write x data (7 components)
            WriteRandomData(x_var_id, start, count, true);

            // Write u data (7 components)
            WriteRandomData(u_var_id, start, count, true);

            // Write v data (6 components)
            WriteRandomData(v_var_id, start, count, false);

            // Write a data (6 components)
            WriteRandomData(a_var_id, start, count, false);

            // Write f data (6 components)
            WriteRandomData(f_var_id, start, count, false);

            if (t % commit_frequency == 0) {
                nc_sync(file_name);  // Flush to disk periodically
            }
        }
    }

    /**
     * @brief Destructor for NetCDFWriter
     */
    ~NetCDFWriter() { nc_close(file_name); }

private:
    int file_name;  ///< NetCDF file ID

    struct VariableIds {
        int x, y, z, w, i, j, k;
    };

    VariableIds x_var_id;  ///< Variable IDs for position
    VariableIds u_var_id;  ///< Variable IDs for displacement
    VariableIds v_var_id;  ///< Variable IDs for velocity
    VariableIds a_var_id;  ///< Variable IDs for acceleration
    VariableIds f_var_id;  ///< Variable IDs for force

    const std::string file_path;  ///< Path to the NetCDF file
    const int num_nodes;          ///< Number of nodes
    const int total_iterations;   ///< Total number of iterations

    std::random_device rd;                 ///< Random device
    std::mt19937 gen;                      ///< Mersenne Twister engine
    std::uniform_real_distribution<> dis;  ///< Uniform distribution

    /**
     * @brief Define variables for a given component (x, u, v, a, or f)
     * @param prefix The prefix for the variable names
     * @param dims The dimension IDs
     * @param var_ids Struct to store the variable IDs
     */
    void DefineVariables(const std::string& prefix, const int dims[2], VariableIds& var_ids) {
        int status;

        if ((status = nc_def_var(file_name, (prefix + "_x").c_str(), NC_DOUBLE, 2, dims, &var_ids.x)
            ) != NC_NOERR ||
            (status = nc_def_var(file_name, (prefix + "_y").c_str(), NC_DOUBLE, 2, dims, &var_ids.y)
            ) != NC_NOERR ||
            (status = nc_def_var(file_name, (prefix + "_z").c_str(), NC_DOUBLE, 2, dims, &var_ids.z)
            ) != NC_NOERR ||
            (status = nc_def_var(file_name, (prefix + "_i").c_str(), NC_DOUBLE, 2, dims, &var_ids.i)
            ) != NC_NOERR ||
            (status = nc_def_var(file_name, (prefix + "_j").c_str(), NC_DOUBLE, 2, dims, &var_ids.j)
            ) != NC_NOERR ||
            (status = nc_def_var(file_name, (prefix + "_k").c_str(), NC_DOUBLE, 2, dims, &var_ids.k)
            ) != NC_NOERR) {
            throw std::runtime_error(nc_strerror(status));
        }

        // Only x and u have w component
        if (prefix == "x" || prefix == "u") {
            if ((status =
                     nc_def_var(file_name, (prefix + "_w").c_str(), NC_DOUBLE, 2, dims, &var_ids.w)
                ) != NC_NOERR) {
                throw std::runtime_error(nc_strerror(status));
            }
        }
    }

    /**
     * @brief Write random data for a set of variables
     * @param var_ids The variable IDs to write to
     * @param start The starting indices
     * @param count The count of values to write
     * @param has_w Whether the variable set includes a w component
     */
    void WriteRandomData(
        const VariableIds& var_ids, const size_t start[2], const size_t count[2], bool has_w
    ) {
        std::vector<double> data(num_nodes);
        int status;

        // Generate and write random data for each component
        // x
        for (int i = 0; i < num_nodes; i++) {
            data[i] = dis(gen);
        }
        if ((status = nc_put_vara_double(file_name, var_ids.x, start, count, data.data())) !=
            NC_NOERR) {
            throw std::runtime_error(nc_strerror(status));
        }

        // y
        for (int i = 0; i < num_nodes; i++) {
            data[i] = dis(gen);
        }
        if ((status = nc_put_vara_double(file_name, var_ids.y, start, count, data.data())) !=
            NC_NOERR) {
            throw std::runtime_error(nc_strerror(status));
        }

        // z
        for (int i = 0; i < num_nodes; i++) {
            data[i] = dis(gen);
        }
        if ((status = nc_put_vara_double(file_name, var_ids.z, start, count, data.data())) !=
            NC_NOERR) {
            throw std::runtime_error(nc_strerror(status));
        }

        // w
        if (has_w) {
            for (int i = 0; i < num_nodes; i++) {
                data[i] = dis(gen);
            }
            if ((status = nc_put_vara_double(file_name, var_ids.w, start, count, data.data())) !=
                NC_NOERR) {
                throw std::runtime_error(nc_strerror(status));
            }
        }

        // i
        for (int i = 0; i < num_nodes; i++) {
            data[i] = dis(gen);
        }
        if ((status = nc_put_vara_double(file_name, var_ids.i, start, count, data.data())) !=
            NC_NOERR) {
            throw std::runtime_error(nc_strerror(status));
        }

        // j
        for (int i = 0; i < num_nodes; i++) {
            data[i] = dis(gen);
        }
        if ((status = nc_put_vara_double(file_name, var_ids.j, start, count, data.data())) !=
            NC_NOERR) {
            throw std::runtime_error(nc_strerror(status));
        }

        // k
        for (int i = 0; i < num_nodes; i++) {
            data[i] = dis(gen);
        }
        if ((status = nc_put_vara_double(file_name, var_ids.k, start, count, data.data())) !=
            NC_NOERR) {
            throw std::runtime_error(nc_strerror(status));
        }
    };
};
