macro(openturbine_enable_cppcheck WARNINGS_AS_ERRORS CPPCHECK_OPTIONS)
  find_program(CPPCHECK cppcheck)
  if(CPPCHECK)
    set(CPPCHECK_TEMPLATE "gcc")
    if(CMAKE_GENERATOR MATCHES ".*Visual Studio.*")
      set(CPPCHECK_TEMPLATE "vs")
    endif()

    # Default cppcheck options if none are provided
    if("${CPPCHECK_OPTIONS}" STREQUAL "")
      set(SUPPRESS_DIR "${CMAKE_CURRENT_BINARY_DIR}/_deps/*.h")
      message(STATUS "CPPCHECK suppressing warnings for directory: ${SUPPRESS_DIR}")
      set(CPPCHECK_OPTIONS
        --enable=style,performance,warning,portability
        --template=${CPPCHECK_TEMPLATE}
        --inline-suppr
        --inconclusive
        --suppress=cppcheckError  # Suppress cppcheck errors
        --suppress=internalAstError # Suppress internal AST errors
        --suppress=unmatchedSuppression # Suppress unmatched suppressions in the code
        --suppress=passedByValue # Suppress passed by value warnings
        --suppress=syntaxError # Suppress syntax errors in the code
        --suppress=preprocessorErrorDirective # Suppress preprocessor error directives
        --suppress=${SUPPRESS_DIR})
    endif()

    if(NOT "${CMAKE_CXX_STANDARD}" STREQUAL "")
      list(APPEND CPPCHECK_OPTIONS --std=c++${CMAKE_CXX_STANDARD})
    endif()

    if(${WARNINGS_AS_ERRORS})
      list(APPEND CPPCHECK_OPTIONS --error-exitcode=2)
    endif()

    set(CMAKE_CXX_CPPCHECK ${CPPCHECK} ${CPPCHECK_OPTIONS})
  else()
    message(WARNING "cppcheck requested but executable not found")
  endif()
endmacro()

macro(openturbine_enable_clang_tidy target WARNINGS_AS_ERRORS)
  find_program(CLANGTIDY clang-tidy)
  if(CLANGTIDY)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
      get_target_property(TARGET_PCH ${target} INTERFACE_PRECOMPILE_HEADERS)
      if(NOT TARGET_PCH)
        get_target_property(TARGET_PCH ${target} PRECOMPILE_HEADERS)
      endif()

      if(TARGET_PCH)
        message(FATAL_ERROR "clang-tidy cannot be used with non-Clang compilers and PCH (Precompiled Headers).")
      endif()
    endif()

    set(CLANG_TIDY_OPTIONS
      ${CLANGTIDY}
      -extra-arg=-Wno-unknown-warning-option # Ignore unknown warning options in Clang-Tidy
      -extra-arg=-Wno-ignored-optimization-argument # Ignore ignored optimization arguments
      -extra-arg=-Wno-unused-command-line-argument # Ignore unused command line arguments
      -p)

    if(NOT "${CMAKE_CXX_STANDARD}" STREQUAL "")
      if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        list(APPEND CLANG_TIDY_OPTIONS -extra-arg=-std=c++${CMAKE_CXX_STANDARD})
      else()
        list(APPEND CLANG_TIDY_OPTIONS -extra-arg=/std:c++${CMAKE_CXX_STANDARD})
      endif()
    endif()

    if(${WARNINGS_AS_ERRORS})
      list(APPEND CLANG_TIDY_OPTIONS -warnings-as-errors=*)
    endif()

    message(STATUS "Setting clang-tidy globally for target: ${target}")
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_OPTIONS})
  else()
    message(WARNING "clang-tidy requested but executable not found")
  endif()
endmacro()
