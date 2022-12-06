
set(Kokkos_DIR "$ENV{Kokkos_ROOT}" CACHE STRING "Kokkos root directory")
find_package(Kokkos REQUIRED)
target_sources(${oturb_exe_name} PRIVATE main.cpp)
target_link_libraries(${oturb_exe_name} PRIVATE Kokkos::kokkos)
