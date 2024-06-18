#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/utilities/controllers/discon.h"
#include "src/vendor/dylib/dylib.hpp"

namespace openturbine::restruct_poc::tests {

TEST(ControllerTest, ClampFunction) {
    // Use dylib to load the dynamic library and get access to the controller functions
    util::dylib lib("./", "DISCON");
    auto clamp = lib.get_function<float(float, float, float)>("clamp");

    // test case 1: v is less than v_min
    float v = 1.0;
    float v_min = 2.0;
    float v_max = 3.0;
    float expected = 2.0;
    auto actual = clamp(v, v_min, v_max);
    EXPECT_FLOAT_EQ(expected, actual);

    // test case 2: v is greater than v_max
    v = 4.0;
    expected = 3.0;
    actual = clamp(v, v_min, v_max);
    EXPECT_FLOAT_EQ(expected, actual);

    // test case 3: v is between v_min and v_max
    v = 2.5;
    expected = 2.5;
    actual = clamp(v, v_min, v_max);
    EXPECT_FLOAT_EQ(expected, actual);
}

TEST(ControllerTest, DisconController) {
    // Test data generated using the following regression test from the
    // OpenFAST/r-test repository:
    // https://github.com/OpenFAST/r-test/tree/main/glue-codes/openfast/5MW_Land_DLL_WTurb
    // at time = 0.0s

    // Use dylib to load the dynamic library and get access to the controller functions
    util::dylib lib("./", "DISCON");
    auto DISCON = lib.get_function<void(float*, int&, char*, char*, char*)>("DISCON");

    float avrSWAP[81] = {0.};
    util::SwapStruct* swap = reinterpret_cast<util::SwapStruct*>(avrSWAP);
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

}  // namespace openturbine::restruct_poc::tests
