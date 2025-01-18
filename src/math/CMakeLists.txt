target_sources(openturbine_library PRIVATE)

install(FILES
        least_squares_fit.hpp
        matrix_operations.hpp
        quaternion_operations.hpp
        vector_operations.hpp
        DESTINATION include/OpenTurbine/math
)
