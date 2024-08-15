# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type specified. Defaulting to 'RelWithDebInfo'.")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)

  # Set the possible values of build type for cmake-gui, ccmake
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")

# Validate the build type if it was specified
elseif(CMAKE_BUILD_TYPE)
  # Make build type case-insensitive and check for supported values
  set(SUPPORTED_BUILD_TYPES Debug Release RelWithDebInfo MinSizeRel)
  string(REPLACE ";" " " SUPPORTED_BUILD_TYPES_STRING "${SUPPORTED_BUILD_TYPES}")

  string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_UPPER)
  string(TOUPPER "${SUPPORTED_BUILD_TYPES}" SUPPORTED_BUILD_TYPES_UPPER)

  if(NOT BUILD_TYPE_UPPER IN_LIST SUPPORTED_BUILD_TYPES_UPPER)
    message(WARNING "Build type ${CMAKE_BUILD_TYPE} is NOT supported - reverting to RelWithDebInfo.")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)
  endif()

endif()

# Print the build type to the console
string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_UPPER)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE_UPPER}")

# Generate compile_commands.json to make it easier to work with clang-based tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enhance error reporting and compiler messages
if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  add_compile_options(-fcolor-diagnostics)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-fdiagnostics-color=always)
else()
  message(STATUS "No colored compiler diagnostics set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
endif()
