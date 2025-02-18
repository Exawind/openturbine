target_sources(openturbine_library PRIVATE)

install(FILES
  write_netcdf_outputs.hpp
  DESTINATION include/OpenTurbine/utilities/outputs/
)
