target_sources(openturbine_library
  PRIVATE
  netcdf_file.cpp
  node_state_writer.cpp
  time_series_writer.cpp
)

install(FILES
  netcdf_file.hpp
  node_state_writer.hpp
  time_series_writer.hpp
  DESTINATION include/OpenTurbine/utilities/netcdf/
)
