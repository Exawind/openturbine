include(cmake/SystemLink.cmake)

macro(openturbine_setup_options)
  # Define project options with default values
  option(OpenTurbine_ENABLE_TESTS "Build Tests" ON)
  option(OpenTurbine_ENABLE_COVERAGE "Enable coverage reporting" OFF)
  option(OpenTurbine_ENABLE_IPO "Enable IPO/LTO (Interprocedural Optimization/Link Time Optimization)" OFF)
  option(OpenTurbine_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)
  option(OpenTurbine_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
  option(OpenTurbine_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
  option(OpenTurbine_ENABLE_SANITIZER_UNDEFINED "Enable undefined behavior sanitizer" OFF)
  option(OpenTurbine_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
  option(OpenTurbine_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
  option(OpenTurbine_ENABLE_UNITY_BUILD "Enable unity builds" OFF)
  option(OpenTurbine_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
  option(OpenTurbine_ENABLE_CPPCHECK "Enable CppCheck analysis" OFF)
  option(OpenTurbine_ENABLE_PCH "Enable precompiled headers" OFF)
  option(OpenTurbine_ENABLE_VTK "Use VTK for visualization" OFF)
  option(OpenTurbine_BUILD_OPENFAST_ADI "Build the OpenFAST ADI external project" OFF)
  option(OpenTurbine_BUILD_ROSCO_CONTROLLER "Build the ROSCO controller external project" ON)
endmacro()

macro(openturbine_global_options)
  # Enable IPO/LTO if the option is set
  if(OpenTurbine_ENABLE_IPO)
    include(cmake/InterproceduralOptimization.cmake)
    openturbine_enable_ipo()
  endif()
endmacro()

macro(openturbine_local_options)
  # Include standard project settings and create interface libraries
  include(cmake/StandardProjectSettings.cmake)
  add_library(openturbine_warnings INTERFACE)
  add_library(openturbine_options INTERFACE)

  # Set compiler warnings
  include(cmake/CompilerWarnings.cmake)
  openturbine_set_project_warnings(
    openturbine_warnings
    ${OpenTurbine_WARNINGS_AS_ERRORS}
    ""
    ""
    ""
  )

  # Enable sanitizers if specified
  include(cmake/Sanitizers.cmake)
  openturbine_enable_sanitizers(
    openturbine_options
    ${OpenTurbine_ENABLE_SANITIZER_ADDRESS}
    ${OpenTurbine_ENABLE_SANITIZER_LEAK}
    ${OpenTurbine_ENABLE_SANITIZER_UNDEFINED}
    ${OpenTurbine_ENABLE_SANITIZER_THREAD}
    ${OpenTurbine_ENABLE_SANITIZER_MEMORY}
  )

  # Set unity build i.e. compile multiple source files into one
  set_target_properties(openturbine_options PROPERTIES UNITY_BUILD ${OpenTurbine_ENABLE_UNITY_BUILD})

  # Enable precompiled headers if specified
  if(OpenTurbine_ENABLE_PCH)
    target_precompile_headers(
      openturbine_options
      INTERFACE
      <vector>
      <string>
      <utility>
    )
  endif()

  # Enable static analysis tools if specified
  include(cmake/StaticAnalyzers.cmake)
  if(OpenTurbine_ENABLE_CLANG_TIDY)
    openturbine_enable_clang_tidy(openturbine_options ${OpenTurbine_WARNINGS_AS_ERRORS})
  endif()

  if(OpenTurbine_ENABLE_CPPCHECK)
    openturbine_enable_cppcheck(${OpenTurbine_WARNINGS_AS_ERRORS} "")
  endif()

  # Enable coverage reporting if specified
  if(OpenTurbine_ENABLE_COVERAGE)
    include(cmake/Tests.cmake)
    openturbine_enable_coverage(openturbine_options)
  endif()
endmacro()
