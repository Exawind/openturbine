target_sources(openturbine_library
  PRIVATE
  blade_interface.cpp
  blade_interface_builder.cpp
)

install(FILES
        blade_interface.hpp
        blade_interface_builder.hpp
        DESTINATION include/OpenTurbine/interfaces/blade/
)
