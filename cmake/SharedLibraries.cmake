function(add_shared_library target_name)
  add_library(${target_name} SHARED)
  set_target_properties(${target_name} PROPERTIES PREFIX "" SUFFIX ".dll")
  # Copy the shared library to the binary directory after building
  add_custom_command(TARGET ${target_name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:${target_name}> ${CMAKE_BINARY_DIR}/$<TARGET_FILE_NAME:${target_name}>
  )
endfunction()

# Define the DISCON shared libraries for the turbine controller
add_shared_library(DISCON)
add_shared_library(DISCON_ROTOR_TEST_CONTROLLER)
