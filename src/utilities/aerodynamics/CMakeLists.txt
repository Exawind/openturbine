target_sources(kynema_library PRIVATE)

install(FILES
        aerodyn_inflow.hpp
        DESTINATION include/Kynema/utilities/aerodynamics/
)
