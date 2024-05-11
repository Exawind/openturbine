
target_sources(
    ${oturb_unit_test_exe_name}
    PRIVATE
    test_beams.cpp
    test_solver.cpp
    test_rotating_beam.cpp
    test_cantilever_beam.cpp
    test_rotor.cpp
    test_math.cpp
    test_utilities.cpp
    test_controller.cpp
)

if(OTURB_ENABLE_VTK)
    target_sources(
        ${oturb_unit_test_exe_name}
        PRIVATE
        vtkout.hpp
    )
endif()
