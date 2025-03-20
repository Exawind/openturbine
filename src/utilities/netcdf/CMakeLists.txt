target_sources(openturbine_library PRIVATE)

install(FILES
  netcdf_file.hpp
  node_state_writer.hpp
  time_series_writer.hpp
  DESTINATION include/OpenTurbine/utilities/netcdf/
)
