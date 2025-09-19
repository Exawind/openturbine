target_sources(kynema_library PRIVATE)

install(FILES
        create_masses.hpp
        mass_element.hpp
        masses.hpp
        masses_input.hpp
        DESTINATION include/Kynema/elements/masses/
)
