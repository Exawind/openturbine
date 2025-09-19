target_sources(kynema_library
  PRIVATE
  turbine_interface.cpp
)

install(FILES
        turbine_interface.hpp
        turbine_interface_builder.hpp
        DESTINATION include/Kynema/interfaces/turbine/
)
