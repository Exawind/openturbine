#=============================================================================
# OpenTurbine Testing
#=============================================================================

# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CTestCustom.cmake ${CMAKE_BINARY_DIR}/CTestCustom.cmake)

if(OTURB_SAVE_GOLDS)
    if("${OTURB_SAVED_GOLDS_DIRECTORY}" STREQUAL "")
        message(FATAL_ERROR "To save gold files, OTURB_SAVED_GOLDS_DIRECTORY must be set and the directory exist")
    else()
        if(EXISTS ${OTURB_SAVED_GOLDS_DIRECTORY})
            set(SAVED_GOLDS_DIRECTORY ${OTURB_SAVED_GOLDS_DIRECTORY}/${CMAKE_SYSTEM_NAME}/${CMAKE_CXX_COMPILER_ID}/${CMAKE_CXX_COMPILER_VERSION})
            message(STATUS "Gold files will be saved to: ${SAVED_GOLDS_DIRECTORY}")
        else()
            message(FATAL_ERROR "Specified directory for saving gold files does not exist: ${OTURB_SAVED_GOLDS_DIRECTORY}")
        endif()
    endif()
endif()

# Have CMake discover the number of cores on the node
include(ProcessorCount)
ProcessorCount(PROCESSES)

#=============================================================================
# Functions for adding tests / Categories of tests
#=============================================================================
macro(setup_test)
    set(CURRENT_TEST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test_files/${TEST_NAME})
    set(CURRENT_TEST_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/test_files/${TEST_NAME})
    # set(PLOT_GOLD ${GOLD_FILES_DIRECTORY}/${TEST_NAME}/plt00010)
    # set(PLOT_TEST ${CURRENT_TEST_BINARY_DIR}/plt00010)
    file(MAKE_DIRECTORY ${CURRENT_TEST_BINARY_DIR})
    file(GLOB TEST_FILES "${CURRENT_TEST_SOURCE_DIR}/*")
    file(COPY ${TEST_FILES} DESTINATION "${CURRENT_TEST_BINARY_DIR}/")
    set(RUNTIME_OPTIONS "time.max_step=10 amr.plot_file=plt time.plot_interval=10 amrex.the_arena_is_managed=0 amrex.abort_on_out_of_gpu_memory=1")
    # if(OTURB_ENABLE_CUDA OR (OTURB_ENABLE_HYPRE AND (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")))
    #   set(FCOMPARE_TOLERANCE "-r 1e-10 --abs_tol 1.0e-12")
    #   set(RUNTIME_OPTIONS "${RUNTIME_OPTIONS} io.skip_outputs=p")
    # endif()
    if(OTURB_SAVE_GOLDS)
        file(MAKE_DIRECTORY ${SAVED_GOLDS_DIRECTORY}/${TEST_NAME})
        set(SAVE_GOLDS_COMMAND "&& cp -R ${PLOT_TEST} ${SAVED_GOLDS_DIRECTORY}/${TEST_NAME}/")
    endif()
endmacro(setup_test)

# Standard regression test
function(add_test_r TEST_NAME)
    setup_test()
    add_test(${TEST_NAME} sh -c "${CMAKE_BINARY_DIR}/${OTURB_exe_name} ${MPIEXEC_POSTFLAGS} ${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.inp ${RUNTIME_OPTIONS} > ${TEST_NAME}.log ${SAVE_GOLDS_COMMAND}")
    # Set properties for test
    set_tests_properties(
        ${TEST_NAME} PROPERTIES
        TIMEOUT 5400
        # PROCESSORS ${TEST_NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "regression"
        ATTACHED_FILES_ON_FAIL "${CURRENT_TEST_BINARY_DIR}/${TEST_NAME}.log"
    )
endfunction(add_test_r)

# Standard unit test
function(add_test_u TEST_NAME)
    setup_test()
    # set(TEST_NP 1)
    add_test(${TEST_NAME} sh -c "${MPI_COMMANDS} ${CMAKE_BINARY_DIR}/${oturb_unit_test_exe_name}")
    set_tests_properties(
        ${TEST_NAME} PROPERTIES
        TIMEOUT 500
        # PROCESSORS ${TEST_NP}
        WORKING_DIRECTORY "${CURRENT_TEST_BINARY_DIR}/"
        LABELS "unit"
    )
endfunction(add_test_u)

#=============================================================================
# Unit tests
#=============================================================================

#=============================================================================
# Regression tests
#=============================================================================
# add_test_r(asdf)

#=============================================================================
# Verification tests
#=============================================================================
# add_test_v(mms "${LIST_OF_GRID_SIZES}")

#=============================================================================
# Performance tests
#=============================================================================
# add_test_p(asdf)
