target_sources(kynema_library
  PRIVATE
  blade_interface.cpp
  blade_interface_builder.cpp
)

install(FILES
        blade_interface.hpp
        blade_interface_builder.hpp
        DESTINATION include/Kynema/interfaces/blade/
)
