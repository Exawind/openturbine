#pragma once

#include <array>

namespace openturbine::util {

static constexpr int kSwapArraySize = 81;

/// Structure defining the swap layout (See Appendix A of Bladed User's Guide)
/// Ref: https://openfast.readthedocs.io/en/main/source/user/servodyn/ExtendedBladedInterface.html
struct ControllerIO {
    int status = 0;    // (1) -->  Status flag set as follows: 0 if this is the first call, 1 for all
                       // subsequent time steps, -1 if this is the final call at the end of the
                       // simulation (-)
    double time = 0.;  // (2) -->  Current time (sec) [t in single precision]
    double dt = 0.;    // (3) -->  Communication interval (sec)
    double pitch_blade1 = 0.;              // (4) -->  Blade 1 pitch angle (rad)
    double pitch_angle_actuator_req = 0.;  // (10)
    double generator_speed = 0.;           // (20) Generator speed (RPM)
    double horizontal_wind_speed = 0.;     // (27) Horizontal hub-heigh wind speed (m/s)
    double pitch_blade2 = 0.;              // (33) Blade 2 pitch
    double pitch_blade3 = 0.;              // (34) Blade 3 pitch
    int generator_contactor_status = 0;    // (35) Generator contactor status: 1=main (high speed)
                                           // variable-speed generator
    int shaft_brake_status = 0;            // (36) Shaft brake status: 0=off
    double demanded_yaw_actuator_torque = 0.;  // (41) Demanded yaw actuator torque
    double pitch_command_1 = 0.;  // (42) Use the command angles of all blades if using individual
                                  // pitch
    double pitch_command_2 = 0.;  // (43) Use the command angles of all blades if using individual
                                  // pitch
    double pitch_command_3 = 0.;  // (44) Use the command angles of all blades if using individual
                                  // pitch
    double pitch_command_collective = 0.;      // (45) Use the command angle of blade 1 if using
                                               // collective pitch
    double demanded_pitch_rate = 0.;           // (46) Demanded pitch rate (Collective pitch)
    double demanded_generator_torque = 0.;     // (47) Demanded generator torque
    double demanded_nacelle_yaw_rate = 0.;     // (48) Demanded nacelle yaw rate
    size_t message_array_size = 0U;            // (49) avcMSG array size
    size_t infile_array_size = 0U;             // (50) avcINFILE array size
    size_t outname_array_size = 0U;            // (51) avcOUTNAME array size
    int pitch_override = 0;                    // (55) Pitch override: 0=yes
    int torque_override = 0;                   // (56) Torque override: 0=yes
    size_t n_blades = 0U;                      // (61) Number of blades
    size_t n_log_variables = 0U;               // (65) Number of variables returned for logging
    double generator_startup_resistance = 0.;  // (72) Generator start-up resistance
    int loads_request = 0;                     // (79) Request for loads: 0=none
    int variable_slip_status = 0;              // (80) Variable slip current status
    int variable_slip_demand = 0;              // (81) Variable slip current demand

    void CopyToSwapArray(std::array<float, kSwapArraySize>& swap_array) const {
        swap_array[0] = static_cast<float>(status);
        swap_array[1] = static_cast<float>(time);
        swap_array[2] = static_cast<float>(dt);
        swap_array[3] = static_cast<float>(pitch_blade1);
        swap_array[9] = static_cast<float>(pitch_angle_actuator_req);
        swap_array[19] = static_cast<float>(generator_speed);
        swap_array[26] = static_cast<float>(horizontal_wind_speed);
        swap_array[32] = static_cast<float>(pitch_blade2);
        swap_array[33] = static_cast<float>(pitch_blade3);
        swap_array[34] = static_cast<float>(generator_contactor_status);
        swap_array[35] = static_cast<float>(shaft_brake_status);
        swap_array[40] = static_cast<float>(demanded_yaw_actuator_torque);
        swap_array[41] = static_cast<float>(pitch_command_1);
        swap_array[42] = static_cast<float>(pitch_command_2);
        swap_array[43] = static_cast<float>(pitch_command_3);
        swap_array[44] = static_cast<float>(pitch_command_collective);
        swap_array[45] = static_cast<float>(demanded_pitch_rate);
        swap_array[46] = static_cast<float>(demanded_generator_torque);
        swap_array[47] = static_cast<float>(demanded_nacelle_yaw_rate);
        swap_array[48] = static_cast<float>(message_array_size);
        swap_array[49] = static_cast<float>(infile_array_size);
        swap_array[50] = static_cast<float>(outname_array_size);
        swap_array[54] = static_cast<float>(pitch_override);
        swap_array[55] = static_cast<float>(torque_override);
        swap_array[60] = static_cast<float>(n_blades);
        swap_array[64] = static_cast<float>(n_log_variables);
        swap_array[71] = static_cast<float>(generator_startup_resistance);
        swap_array[78] = static_cast<float>(loads_request);
        swap_array[79] = static_cast<float>(variable_slip_status);
        swap_array[80] = static_cast<float>(variable_slip_demand);
    }

    void CopyFromSwapArray(const std::array<float, kSwapArraySize>& swap_array) {
        status = static_cast<int>(swap_array[0]);
        time = static_cast<double>(swap_array[1]);
        dt = static_cast<double>(swap_array[2]);
        pitch_blade1 = static_cast<double>(swap_array[3]);
        pitch_angle_actuator_req = static_cast<double>(swap_array[9]);
        generator_speed = static_cast<double>(swap_array[19]);
        horizontal_wind_speed = static_cast<double>(swap_array[26]);
        pitch_blade2 = static_cast<double>(swap_array[32]);
        pitch_blade3 = static_cast<double>(swap_array[33]);
        generator_contactor_status = static_cast<int>(swap_array[34]);
        shaft_brake_status = static_cast<int>(swap_array[35]);
        demanded_yaw_actuator_torque = static_cast<double>(swap_array[40]);
        pitch_command_1 = static_cast<double>(swap_array[41]);
        pitch_command_2 = static_cast<double>(swap_array[42]);
        pitch_command_3 = static_cast<double>(swap_array[43]);
        pitch_command_collective = static_cast<double>(swap_array[44]);
        demanded_pitch_rate = static_cast<double>(swap_array[45]);
        demanded_generator_torque = static_cast<double>(swap_array[46]);
        demanded_nacelle_yaw_rate = static_cast<double>(swap_array[47]);
        message_array_size = static_cast<size_t>(swap_array[48]);
        infile_array_size = static_cast<size_t>(swap_array[49]);
        outname_array_size = static_cast<size_t>(swap_array[50]);
        pitch_override = static_cast<int>(swap_array[54]);
        torque_override = static_cast<int>(swap_array[55]);
        n_blades = static_cast<size_t>(swap_array[60]);
        n_log_variables = static_cast<size_t>(swap_array[64]);
        generator_startup_resistance = static_cast<double>(swap_array[71]);
        loads_request = static_cast<int>(swap_array[78]);
        variable_slip_status = static_cast<int>(swap_array[79]);
        variable_slip_demand = static_cast<int>(swap_array[80]);
    }
};

}  // namespace openturbine::util
