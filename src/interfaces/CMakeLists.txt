add_subdirectory(cfd)
add_subdirectory(blade)
add_subdirectory(components)
add_subdirectory(turbine)

target_sources(openturbine_library
  PRIVATE
  node_data.cpp
  outputs.cpp
)

install(FILES
	constraint_data.hpp
        host_state.hpp
        node_data.hpp
        outputs.hpp
        DESTINATION include/OpenTurbine/interfaces/
)
