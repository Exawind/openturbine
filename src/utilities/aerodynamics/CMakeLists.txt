target_sources(openturbine_library PRIVATE)

install(FILES
        aerodyn_inflow.hpp
        DESTINATION include/OpenTurbine/utilities/aerodynamics/
)
