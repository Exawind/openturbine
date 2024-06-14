#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/utilities/controllers/discon_cpp.h"
#include "src/utilities/dylib.hpp"

namespace openturbine::restruct_poc::tests {

TEST(ControllerTest, ClampFunction) {
    // Use dylib to load the dynamic library and get access to the controller functions
    // util::dylib lib("./", "DISCON");
    // auto clamp = lib.get_function<float(float, float, float)>("clamp");

    // test case 1: v is less than v_min
    float v = 1.0;
    float v_min = 2.0;
    float v_max = 3.0;
    float expected = 2.0;
    auto actual = openturbine::util::clamp<float>(v, v_min, v_max);
    EXPECT_FLOAT_EQ(expected, actual);

    // test case 2: v is greater than v_max
    v = 4.0;
    expected = 3.0;
    actual = util::clamp<float>(v, v_min, v_max);
    EXPECT_FLOAT_EQ(expected, actual);

    // test case 3: v is between v_min and v_max
    v = 2.5;
    expected = 2.5;
    actual = util::clamp<float>(v, v_min, v_max);
    EXPECT_FLOAT_EQ(expected, actual);
}

TEST(ControllerTest, DisconController) {
    // Test data generated using the following regression test from the
    // OpenFAST/r-test repository:
    // https://github.com/OpenFAST/r-test/tree/main/glue-codes/openfast/5MW_Land_DLL_WTurb
    // at time = 0.0s

    // Use dylib to load the dynamic library and get access to the controller functions
    // util::dylib lib("./", "DISCON");
    // auto DISCON = lib.get_function<void(float*, int&, char*, char*, char*)>("DISCON");

    float avrSWAP[81] = {0.};
    avrSWAP[0] = 0.;           // Status
    avrSWAP[1] = 0.;           // Time
    avrSWAP[3] = 0.;           // BlPitch1
    avrSWAP[9] = 0.;           // PitchAngleActuatorReq
    avrSWAP[19] = 122.909576;  // GenSpeed
    avrSWAP[26] = 11.9900799;  // HorWindV
    avrSWAP[32] = 0.;          // BlPitch2
    avrSWAP[33] = 0.;          // BlPitch3
    avrSWAP[34] = 1.;          // GeneratorContactorStatus
    avrSWAP[35] = 0.;          // ShaftBrakeStatus
    avrSWAP[40] = 0.;          // DemandedYawActuatorTorque
    avrSWAP[41] = 0.;          // PitchCom1
    avrSWAP[42] = 0.;          // PitchCom2
    avrSWAP[43] = 0.;          // PitchCom3
    avrSWAP[44] = 0.;          // PitchComCol
    avrSWAP[45] = 0.;          // DemandedPitchRate
    avrSWAP[46] = 0.;          // DemandedGeneratorTorque
    avrSWAP[47] = 0.;          // DemandedNacelleYawRate
    avrSWAP[48] = 3.;          // msg_size
    avrSWAP[49] = 82.;         // infile_size
    avrSWAP[50] = 96.;         // outname_size
    avrSWAP[54] = 0.;          // PitchOverride
    avrSWAP[55] = 0.;          // TorqueOverride
    avrSWAP[60] = 3.;          // NumBl
    avrSWAP[64] = 0.;          // NumVar
    avrSWAP[71] = 0.;          // GeneratorStartResistance
    avrSWAP[78] = 0.;          // LoadsReq
    avrSWAP[79] = 0.;          // VariableSlipStatus
    avrSWAP[80] = 0.;          // VariableSlipDemand

    // Call DISCON and expect the following outputs
    int aviFAIL = 0;
    char in_file[] = "in_file";
    char out_name[] = "out_name";
    char msg[] = "msg";
    openturbine::util::DISCON(avrSWAP, aviFAIL, in_file, out_name, msg);

    EXPECT_FLOAT_EQ(avrSWAP[34], 1.);          // GeneratorContactorStatus
    EXPECT_FLOAT_EQ(avrSWAP[35], 0.);          // ShaftBrakeStatus
    EXPECT_FLOAT_EQ(avrSWAP[40], 0.);          // DemandedYawActuatorTorque
    EXPECT_FLOAT_EQ(avrSWAP[44], 0.);          // PitchComCol
    EXPECT_FLOAT_EQ(avrSWAP[46], 43093.5508);  // DemandedGeneratorTorque
    EXPECT_FLOAT_EQ(avrSWAP[47], 0.);          // DemandedNacelleYawRate
}

}  // namespace openturbine::restruct_poc::tests
