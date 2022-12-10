
set(Kokkos_DIR "$ENV{Kokkos_ROOT}" CACHE STRING "Kokkos root directory")
find_package(Kokkos REQUIRED)
target_sources(${oturb_exe_name} PRIVATE main.cpp)
target_link_libraries(${oturb_exe_name} PRIVATE Kokkos::kokkos)

# Generate OpenTurbine version header
configure_file("${CMAKE_SOURCE_DIR}/cmake/OpenTurbineVersion.H.in"
    "${CMAKE_CURRENT_BINARY_DIR}/OpenTurbineVersion.H" @ONLY)

target_include_directories(${oturb_exe_name} PRIVATE ${PROJECT_BINARY_DIR})
