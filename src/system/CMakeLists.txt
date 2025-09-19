target_sources(kynema_library PRIVATE)

install(FILES
        calculate_tangent_operator.hpp
        DESTINATION include/Kynema/system/
)

add_subdirectory(beams)
add_subdirectory(masses)
add_subdirectory(springs)
