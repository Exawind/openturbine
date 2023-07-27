target_sources(${oturb_lib_name}
    PRIVATE
    #C++
    # diagnostics.cpp
    # io.cpp
    generalized_alpha_time_integrator.cpp
    heavy_top.cpp
    quaternion.cpp
    solver.cpp
    state.cpp
    time_integrator.cpp
    time_stepper.cpp
    utilities.cpp
    vector.cpp
    # IOManager.cpp
)
