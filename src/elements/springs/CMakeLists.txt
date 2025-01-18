target_sources(openturbine_library PRIVATE)

install(FILES
        create_springs.hpp
        spring_element.hpp
        springs.hpp
        springs_input.hpp
        DESTINATION include/OpenTurbine/elements/springs/
)
