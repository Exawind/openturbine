#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/utilities/controllers/discon.hpp"
#include "src/utilities/controllers/turbine_controller.hpp"
#include "src/vendor/dylib/dylib.hpp"

namespace openturbine::restruct_poc::tests {

TEST(ControllerTest, DisconController) {
    // Test data generated using the following regression test from the
    // OpenFAST/r-test repository:
    // https://github.com/OpenFAST/r-test/tree/main/glue-codes/openfast/5MW_Land_DLL_WTurb
    // at time = 0.0s

    // Use dylib to load the dynamic library and get access to the controller functions
    util::dylib lib("./DISCON.dll", util::dylib::no_filename_decorations);
    auto DISCON = lib.get_function<void(float*, int&, char*, char*, char*)>("DISCON");

    float avrSWAP[81] = {0.};
    util::ControllerIO* swap = reinterpret_cast<util::ControllerIO*>(avrSWAP);
    swap->status = 0.;
    swap->time = 0.;
    swap->pitch_blade1 = 0.;
    swap->pitch_angle_actuator_req = 0.;
    swap->generator_speed = 122.909576;
    swap->horizontal_wind_speed = 11.9900799;
    swap->pitch_blade2 = 0.;
    swap->pitch_blade3 = 0.;
    swap->generator_contactor_status = 1.;
    swap->shaft_brake_status = 0.;
    swap->demanded_yaw_actuator_torque = 0.;
    swap->pitch_command_1 = 0.;
    swap->pitch_command_2 = 0.;
    swap->pitch_command_3 = 0.;
    swap->pitch_command_collective = 0.;
    swap->demanded_pitch_rate = 0.;
    swap->demanded_generator_torque = 0.;
    swap->demanded_nacelle_yaw_rate = 0.;
    swap->message_array_size = 3.;
    swap->infile_array_size = 82.;
    swap->outname_array_size = 96.;
    swap->pitch_override = 0.;
    swap->torque_override = 0.;
    swap->n_blades = 3.;
    swap->n_log_variables = 0.;
    swap->generator_startup_resistance = 0.;
    swap->loads_request = 0.;
    swap->variable_slip_status = 0.;
    swap->variable_slip_demand = 0.;

    // Call DISCON and expect the following outputs
    int aviFAIL = 0;
    char in_file[] = "in_file";
    char out_name[] = "out_name";
    char msg[] = "msg";
    DISCON(avrSWAP, aviFAIL, in_file, out_name, msg);

    EXPECT_FLOAT_EQ(avrSWAP[34], 1.);          // GeneratorContactorStatus
    EXPECT_FLOAT_EQ(avrSWAP[35], 0.);          // ShaftBrakeStatus
    EXPECT_FLOAT_EQ(avrSWAP[40], 0.);          // DemandedYawActuatorTorque
    EXPECT_FLOAT_EQ(avrSWAP[44], 0.);          // PitchComCol
    EXPECT_FLOAT_EQ(avrSWAP[46], 43093.5508);  // DemandedGeneratorTorque
    EXPECT_FLOAT_EQ(avrSWAP[47], 0.);          // DemandedNacelleYawRate
}

TEST(ControllerTest, TurbineController) {
    // Get a handle to the controller function via the TurbineController class and use it to
    // calculate the controller outputs
    std::string shared_lib_path = "./DISCON.dll";
    std::string controller_function_name = "DISCON";

    util::TurbineController controller(shared_lib_path, controller_function_name);

    controller.io->status = 0.;
    controller.io->time = 0.;
    controller.io->pitch_blade1 = 0.;
    controller.io->pitch_angle_actuator_req = 0.;
    controller.io->generator_speed = 122.909576;
    controller.io->horizontal_wind_speed = 11.9900799;
    controller.io->pitch_blade2 = 0.;
    controller.io->pitch_blade3 = 0.;
    controller.io->generator_contactor_status = 1.;
    controller.io->shaft_brake_status = 0.;
    controller.io->demanded_yaw_actuator_torque = 0.;
    controller.io->pitch_command_1 = 0.;
    controller.io->pitch_command_2 = 0.;
    controller.io->pitch_command_3 = 0.;
    controller.io->pitch_command_collective = 0.;
    controller.io->demanded_pitch_rate = 0.;
    controller.io->demanded_generator_torque = 0.;
    controller.io->demanded_nacelle_yaw_rate = 0.;
    controller.io->message_array_size = 3.;
    controller.io->infile_array_size = 82.;
    controller.io->outname_array_size = 96.;
    controller.io->pitch_override = 0.;
    controller.io->torque_override = 0.;
    controller.io->n_blades = 3.;
    controller.io->n_log_variables = 0.;
    controller.io->generator_startup_resistance = 0.;
    controller.io->loads_request = 0.;
    controller.io->variable_slip_status = 0.;
    controller.io->variable_slip_demand = 0.;

    controller.CallController();

    EXPECT_FLOAT_EQ(controller.io->generator_contactor_status, 1.);    // GeneratorContactorStatus
    EXPECT_FLOAT_EQ(controller.io->shaft_brake_status, 0.);            // ShaftBrakeStatus
    EXPECT_FLOAT_EQ(controller.io->demanded_yaw_actuator_torque, 0.);  // DemandedYawActuatorTorque
    EXPECT_FLOAT_EQ(controller.io->pitch_command_collective, 0.);      // PitchComCol
    EXPECT_FLOAT_EQ(
        controller.io->demanded_generator_torque, 43093.5508
    );                                                              // DemandedGeneratorTorque
    EXPECT_FLOAT_EQ(controller.io->demanded_nacelle_yaw_rate, 0.);  // DemandedNacelleYawRate
}

TEST(ControllerTest, TurbineControllerException) {
    // Test case 1: invalid shared library path
    std::string shared_lib_path = "./INVALID.dll";
    std::string controller_function_name = "DISCON";

    EXPECT_THROW(
        util::TurbineController controller(shared_lib_path, controller_function_name),
        std::runtime_error
    );

    // Test case 2: invalid controller function name
    shared_lib_path = "./DISCON.dll";
    controller_function_name = "INVALID";

    EXPECT_THROW(
        util::TurbineController controller(shared_lib_path, controller_function_name),
        std::runtime_error
    );
}

}  // namespace openturbine::restruct_poc::tests
