target_sources(openturbine_library
  PRIVATE
  beam.cpp
  beam_builder.cpp
  solution_builder.cpp
  turbine.cpp
  turbine_builder.cpp
)

install(FILES
        beam.hpp
        beam_builder.hpp
        beam_input.hpp
        controller_input.hpp
        solution_builder.hpp
        solution_input.hpp
        turbine.hpp
        turbine_builder.hpp
        turbine_input.hpp
        DESTINATION include/OpenTurbine/interfaces/components/
)
