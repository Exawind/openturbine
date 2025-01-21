target_sources(openturbine_library PRIVATE)

install(FILES
        elements.hpp
        DESTINATION include/OpenTurbine/elements
)

add_subdirectory(beams)
add_subdirectory(masses)
add_subdirectory(springs)
