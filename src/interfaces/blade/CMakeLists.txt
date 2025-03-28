target_sources(openturbine_library 
  PRIVATE
  blade_interface.cpp
)

install(FILES
        interface.hpp
        interface_builder.hpp
        DESTINATION include/OpenTurbine/interfaces/blade/
)
