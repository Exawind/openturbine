include(cmake/SystemLink.cmake)

macro(openturbine_setup_options)
  option(OpenTurbine_ENABLE_TESTS "Build Tests" ON)
  option(OpenTurbine_ENABLE_COVERAGE "Enable coverage reporting" OFF)
  option(OpenTurbine_ENABLE_IPO "Enable IPO/LTO" OFF)
  option(OpenTurbine_WARNINGS_AS_ERRORS "Treat Warnings As Errors" ON)
  option(OpenTurbine_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
  option(OpenTurbine_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
  option(OpenTurbine_ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" OFF)
  option(OpenTurbine_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
  option(OpenTurbine_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
  option(OpenTurbine_ENABLE_UNITY_BUILD "Enable unity builds" OFF)
  option(OpenTurbine_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
  option(OpenTurbine_ENABLE_CPPCHECK "Enable CppCheck analysis" OFF)
  option(OpenTurbine_ENABLE_PCH "Enable precompiled headers" OFF)
  option(OpenTurbine_ENABLE_VTK "Use VTK for visualization" OFF)
endmacro()

macro(openturbine_global_options)
  if(OpenTurbine_ENABLE_IPO)
    include(cmake/InterproceduralOptimization.cmake)
    openturbine_enable_ipo()
  endif()
endmacro()

macro(openturbine_local_options)
  include(cmake/StandardProjectSettings.cmake)
  add_library(openturbine_warnings INTERFACE)
  add_library(openturbine_options INTERFACE)

  include(cmake/CompilerWarnings.cmake)
  openturbine_set_project_warnings(
    openturbine_warnings
    ${OpenTurbine_WARNINGS_AS_ERRORS}
    ""
    ""
    ""
  )

  include(cmake/Sanitizers.cmake)
  openturbine_enable_sanitizers(
    openturbine_options
    ${OpenTurbine_ENABLE_SANITIZER_ADDRESS}
    ${OpenTurbine_ENABLE_SANITIZER_LEAK}
    ${OpenTurbine_ENABLE_SANITIZER_UNDEFINED}
    ${OpenTurbine_ENABLE_SANITIZER_THREAD}
    ${OpenTurbine_ENABLE_SANITIZER_MEMORY}
  )

  set_target_properties(openturbine_options PROPERTIES UNITY_BUILD ${OpenTurbine_ENABLE_UNITY_BUILD})

  if(OpenTurbine_ENABLE_PCH)
    target_precompile_headers(
      openturbine_options
      INTERFACE
      <vector>
      <string>
      <utility>)
  endif()

  include(cmake/StaticAnalyzers.cmake)
  if(OpenTurbine_ENABLE_CLANG_TIDY)
    openturbine_enable_clang_tidy(openturbine_options ${OpenTurbine_WARNINGS_AS_ERRORS})
  endif()

  if(OpenTurbine_ENABLE_CPPCHECK)
    openturbine_enable_cppcheck(${OpenTurbine_WARNINGS_AS_ERRORS} "")
  endif()

  if(OpenTurbine_ENABLE_COVERAGE)
    include(cmake/Tests.cmake)
    openturbine_enable_coverage(openturbine_options)
  endif()
endmacro()