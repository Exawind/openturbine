add_subdirectory(cfd)
add_subdirectory(blade)
add_subdirectory(components)
add_subdirectory(turbine)

install(FILES
        host_state.hpp
        node_data.hpp
        outputs.hpp
        DESTINATION include/OpenTurbine/interfaces/
)
