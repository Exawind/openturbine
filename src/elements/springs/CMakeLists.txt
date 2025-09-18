target_sources(kynema_library PRIVATE)

install(FILES
        create_springs.hpp
        spring_element.hpp
        springs.hpp
        springs_input.hpp
        DESTINATION include/Kynema/elements/springs/
)
