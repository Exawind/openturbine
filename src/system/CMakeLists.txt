target_sources(openturbine_library PRIVATE)

install(FILES
        calculate_tangent_operator.hpp
        DESTINATION include/OpenTurbine/system/
)

add_subdirectory(beams)
add_subdirectory(masses)
add_subdirectory(springs)
