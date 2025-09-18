target_sources(kynema_library PRIVATE)

install(FILES
        elements.hpp
        DESTINATION include/Kynema/elements
)

add_subdirectory(beams)
add_subdirectory(masses)
add_subdirectory(springs)
