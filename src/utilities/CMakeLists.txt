target_sources(openturbine_library
    PRIVATE
    debug_utils.cpp
    log.cpp
    controllers/turbine_controller.cpp
)

target_sources(DISCON
  PRIVATE 
  controllers/discon.cpp)

target_sources(DISCON_ROTOR_TEST_CONTROLLER
  PRIVATE
  controllers/discon_rotor_test_controller.cpp)