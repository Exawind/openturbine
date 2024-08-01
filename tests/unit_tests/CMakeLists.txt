
# Add subdirectories for unit tests
#add_subdirectory(gebt_poc)
#add_subdirectory(gen_alpha_poc)
add_subdirectory(restruct_poc)

target_sources(
    ${oturb_unit_test_exe_name}
    PRIVATE
    utest_main.cpp
    test_config.cpp
)

target_compile_options(
    ${oturb_unit_test_exe_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${OTURB_CXX_FLAGS}>
)
target_include_directories(${oturb_unit_test_exe_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${oturb_unit_test_exe_name} PRIVATE ${PROJECT_BINARY_DIR})

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    set(FS_LIB stdc++fs)
  endif()
endif()

# Link our unit test executable with GoogleTest
find_package(GTest REQUIRED)
target_link_libraries(${oturb_unit_test_exe_name} PRIVATE ${FS_LIB} GTest::gtest GTest::gtest_main)

# Link to OpenTurbine test targets
target_link_libraries(${oturb_unit_test_exe_name} PRIVATE ${oturb_lib_name})

# Link Kokkos to test target
find_package(KokkosKernels REQUIRED)
find_package(Amesos2 REQUIRED)
target_link_libraries(${oturb_unit_test_exe_name} PRIVATE
  KokkosKernels::kokkoskernels
  Amesos2::amesos2
)

# Link VTK to test target
if (OTURB_ENABLE_VTK)
  target_link_libraries(${oturb_unit_test_exe_name} PRIVATE
    VTK::IOXML
  )
endif()

# Define what we want to be installed during a make install
install(TARGETS ${oturb_unit_test_exe_name}
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

