
if(OTURB_ENABLE_ALL_WARNINGS)

    # GCC, Clang, and Intel seem to accept these
    list(APPEND OTURB_CXX_FLAGS "-Wall" "-Wextra" "-pedantic")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        # Intel always reports some diagnostics we don't necessarily care about
        list(APPEND
            OTURB_CXX_FLAGS
            "-diag-disable:11074,11076,15335"
        )
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7.0)
            list(APPEND
                OTURB_CXX_FLAGS
                "-faligned-new"
                "-Wunreachable-code"
                "-Wnull-dereference"
                "-Wfloat-conversion"
                "-Wshadow"
                "-Woverloaded-virtual"
            )
        endif()
    endif()

endif()

# Add our extra flags according to language
separate_arguments(OTURB_CXX_FLAGS)
target_compile_options(
    ${oturb_lib_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${OTURB_CXX_FLAGS}>
)

# Building on CUDA requires additional considerations
# if (OTURB_ENABLE_CUDA)
#     set_target_properties(
#         ${oturb_lib_name} PROPERTIES
#         CUDA_SEPARABLE_COMPILATION ON
#     )
# endif()

# if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
#     if ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") AND OTURB_ENABLE_FPE_TRAP_FOR_TESTS)
#         target_compile_options(
#             ${oturb_lib_name} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-ffp-exception-behavior=maytrap>
#         )
#     endif()
#     target_compile_options(
#         ${oturb_lib_name} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-Wno-pass-failed>
#     )
# endif()
