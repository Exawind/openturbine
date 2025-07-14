#include "netcdf_file.hpp"

namespace openturbine::util {
NetCDFFile::NetCDFFile(const std::string& file_path, bool create) {
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

NetCDFFile::~NetCDFFile() {
    // Close if NetCDF file ID is valid
    if (netcdf_id_ != -1) {
        nc_close(netcdf_id_);
        netcdf_id_ = -1;
    }
}

int NetCDFFile::AddDimension(const std::string& name, size_t length) const {
    int dim_id{-1};
    check_netCDF_error(
        nc_def_dim(netcdf_id_, name.c_str(), length, &dim_id), "Failed to create dimension " + name
    );
    return dim_id;
}

template <>
int NetCDFFile::AddVariable<float>(const std::string& name, const std::vector<int>& dim_ids) const {
    int var_id{-1};
    check_netCDF_error(
        nc_def_var(
            netcdf_id_, name.c_str(), NC_FLOAT, static_cast<int>(dim_ids.size()), dim_ids.data(),
            &var_id
        ),
        "Failed to create variable " + name
    );
    return var_id;
}

template <>
int NetCDFFile::AddVariable<double>(const std::string& name, const std::vector<int>& dim_ids) const {
    int var_id{-1};
    check_netCDF_error(
        nc_def_var(
            netcdf_id_, name.c_str(), NC_DOUBLE, static_cast<int>(dim_ids.size()), dim_ids.data(),
            &var_id
        ),
        "Failed to create variable " + name
    );
    return var_id;
}

template <>
int NetCDFFile::AddVariable<int>(const std::string& name, const std::vector<int>& dim_ids) const {
    int var_id{-1};
    check_netCDF_error(
        nc_def_var(
            netcdf_id_, name.c_str(), NC_INT, static_cast<int>(dim_ids.size()), dim_ids.data(),
            &var_id
        ),
        "Failed to create variable " + name
    );
    return var_id;
}

template <>
int NetCDFFile::AddVariable<std::string>(const std::string& name, const std::vector<int>& dim_ids)
    const {
    int var_id{-1};
    check_netCDF_error(
        nc_def_var(
            netcdf_id_, name.c_str(), NC_STRING, static_cast<int>(dim_ids.size()), dim_ids.data(),
            &var_id
        ),
        "Failed to create variable " + name
    );
    return var_id;
}

void NetCDFFile::AddAttribute(const std::string& var_name, const std::string& attr_name, float value)
    const {
    check_netCDF_error(
        nc_put_att(
            netcdf_id_, this->GetVariableId(var_name), attr_name.c_str(), NC_FLOAT, 1, &value
        ),
        "Failed to write attribute " + attr_name
    );
}

void NetCDFFile::AddAttribute(
    const std::string& var_name, const std::string& attr_name, double value
) const {
    check_netCDF_error(
        nc_put_att(
            netcdf_id_, this->GetVariableId(var_name), attr_name.c_str(), NC_DOUBLE, 1, &value
        ),
        "Failed to write attribute " + attr_name
    );
}

void NetCDFFile::AddAttribute(const std::string& var_name, const std::string& attr_name, int value)
    const {
    check_netCDF_error(
        nc_put_att(netcdf_id_, this->GetVariableId(var_name), attr_name.c_str(), NC_INT, 1, &value),
        "Failed to write attribute " + attr_name
    );
}

void NetCDFFile::AddAttribute(
    const std::string& var_name, const std::string& attr_name, const std::string& value
) const {
    check_netCDF_error(
        nc_put_att_text(
            netcdf_id_, this->GetVariableId(var_name), attr_name.c_str(), value.length(),
            value.c_str()
        ),
        "Failed to write attribute " + attr_name
    );
}

void NetCDFFile::WriteVariable(const std::string& name, const std::vector<float>& data) const {
    check_netCDF_error(
        nc_put_var(netcdf_id_, this->GetVariableId(name), data.data()),
        "Failed to write variable " + name
    );
}

void NetCDFFile::WriteVariable(const std::string& name, const std::vector<double>& data) const {
    check_netCDF_error(
        nc_put_var(netcdf_id_, this->GetVariableId(name), data.data()),
        "Failed to write variable " + name
    );
}

void NetCDFFile::WriteVariable(const std::string& name, const std::vector<int>& data) const {
    check_netCDF_error(
        nc_put_var(netcdf_id_, this->GetVariableId(name), data.data()),
        "Failed to write variable " + name
    );
}

void NetCDFFile::WriteVariable(const std::string& name, const std::vector<std::string>& data) const {
    std::vector<const char*> c_strs;
    c_strs.resize(data.size());
    std::transform(
        std::cbegin(data), std::cend(data), std::begin(c_strs),
        [](const std::string& str) {
            return str.c_str();
        }
    );
    check_netCDF_error(
        nc_put_var_string(netcdf_id_, this->GetVariableId(name), c_strs.data()),
        "Failed to write string variable " + name
    );
}

void NetCDFFile::WriteVariableAt(
    const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
    const std::vector<float>& data
) const {
    check_netCDF_error(
        nc_put_vara(netcdf_id_, this->GetVariableId(name), start.data(), count.data(), data.data()),
        "Failed to write variable " + name
    );
}

void NetCDFFile::WriteVariableAt(
    const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
    const std::vector<double>& data
) const {
    check_netCDF_error(
        nc_put_vara(netcdf_id_, this->GetVariableId(name), start.data(), count.data(), data.data()),
        "Failed to write variable " + name
    );
}

void NetCDFFile::WriteVariableAt(
    const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
    const std::vector<int>& data
) const {
    check_netCDF_error(
        nc_put_vara(netcdf_id_, this->GetVariableId(name), start.data(), count.data(), data.data()),
        "Failed to write variable " + name
    );
}

void NetCDFFile::WriteVariableAt(
    const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
    const std::vector<std::string>& data
) const {
    std::vector<const char*> c_strs;
    c_strs.resize(data.size());
    std::transform(
        std::cbegin(data), std::cend(data), std::begin(c_strs),
        [](const std::string& str) {
            return str.c_str();
        }
    );
    check_netCDF_error(
        nc_put_vara_string(
            netcdf_id_, this->GetVariableId(name), start.data(), count.data(), c_strs.data()
        ),
        "Failed to write string variable " + name
    );
}

int NetCDFFile::GetNetCDFId() const {
    return netcdf_id_;
}

int NetCDFFile::GetDimensionId(const std::string& name) const {
    int dim_id{-1};
    check_netCDF_error(
        nc_inq_dimid(netcdf_id_, name.c_str(), &dim_id), "Failed to get dimension ID for " + name
    );
    return dim_id;
}

int NetCDFFile::GetVariableId(const std::string& name) const {
    int var_id{-1};
    check_netCDF_error(
        nc_inq_varid(netcdf_id_, name.c_str(), &var_id), "Failed to get variable ID for " + name
    );
    return var_id;
}

size_t NetCDFFile::GetNumberOfDimensions(const std::string& var_name) const {
    const int var_id = GetVariableId(var_name);
    int num_dims{0};
    check_netCDF_error(
        nc_inq_varndims(netcdf_id_, var_id, &num_dims),
        "Failed to get number of dimensions for variable " + var_name
    );
    return static_cast<size_t>(num_dims);
}

size_t NetCDFFile::GetDimensionLength(int dim_id) const {
    size_t length{0};
    check_netCDF_error(nc_inq_dimlen(netcdf_id_, dim_id, &length), "Failed to get dimension length");
    return length;
}

size_t NetCDFFile::GetDimensionLength(const std::string& name) const {
    return GetDimensionLength(GetDimensionId(name));
}

std::vector<size_t> NetCDFFile::GetShape(const std::string& var_name) const {
    const int var_id = GetVariableId(var_name);
    const size_t num_dims = GetNumberOfDimensions(var_name);

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

void NetCDFFile::ReadVariable(const std::string& name, float* data) const {
    check_netCDF_error(
        nc_get_var_float(netcdf_id_, this->GetVariableId(name), data),
        "Failed to read float variable " + name
    );
}

void NetCDFFile::ReadVariable(const std::string& name, double* data) const {
    check_netCDF_error(
        nc_get_var_double(netcdf_id_, this->GetVariableId(name), data),
        "Failed to read double variable " + name
    );
}

void NetCDFFile::ReadVariable(const std::string& name, int* data) const {
    check_netCDF_error(
        nc_get_var_int(netcdf_id_, this->GetVariableId(name), data),
        "Failed to read int variable " + name
    );
}

void NetCDFFile::ReadVariableAt(
    const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
    float* data
) const {
    check_netCDF_error(
        nc_get_vara_float(netcdf_id_, this->GetVariableId(name), start.data(), count.data(), data),
        "Failed to read float variable " + name
    );
}

void NetCDFFile::ReadVariableAt(
    const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
    double* data
) const {
    check_netCDF_error(
        nc_get_vara_double(netcdf_id_, this->GetVariableId(name), start.data(), count.data(), data),
        "Failed to read double variable " + name
    );
}

void NetCDFFile::ReadVariableAt(
    const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
    int* data
) const {
    check_netCDF_error(
        nc_get_vara_int(netcdf_id_, this->GetVariableId(name), start.data(), count.data(), data),
        "Failed to read int variable " + name
    );
}

void NetCDFFile::ReadVariableWithStride(
    const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
    const std::vector<ptrdiff_t>& stride, float* data
) const {
    check_netCDF_error(
        nc_get_vars_float(
            netcdf_id_, this->GetVariableId(name), start.data(), count.data(), stride.data(), data
        ),
        "Failed to read float variable with stride " + name
    );
}

void NetCDFFile::ReadVariableWithStride(
    const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
    const std::vector<ptrdiff_t>& stride, double* data
) const {
    check_netCDF_error(
        nc_get_vars_double(
            netcdf_id_, this->GetVariableId(name), start.data(), count.data(), stride.data(), data
        ),
        "Failed to read double variable with stride " + name
    );
}

void NetCDFFile::ReadVariableWithStride(
    const std::string& name, const std::vector<size_t>& start, const std::vector<size_t>& count,
    const std::vector<ptrdiff_t>& stride, int* data
) const {
    check_netCDF_error(
        nc_get_vars_int(
            netcdf_id_, this->GetVariableId(name), start.data(), count.data(), stride.data(), data
        ),
        "Failed to read int variable with stride " + name
    );
}
}  // namespace openturbine::util
