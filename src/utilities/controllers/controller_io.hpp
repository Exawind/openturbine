#pragma once

namespace openturbine::util {

static constexpr int kSwapArraySize = 81;

/// Structure defining the swap layout (See Appendix A of Bladed User's Guide)
/// Ref: https://openfast.readthedocs.io/en/main/source/user/servodyn/ExtendedBladedInterface.html
struct ControllerIO {
    float status;  // (1) -->  Status flag set as follows: 0 if this is the first call, 1 for all
                   // subsequent time steps, -1 if this is the final call at the end of the
                   // simulation (-)
    float time;    // (2) -->  Current time (sec) [t in single precision]
    float dt;      // (3) -->  Communication interval (sec)
    float pitch_blade1;  // (4) -->  Blade 1 pitch angle (rad)
    float unused_5;      // (5) Below-rated pitch angle set-point (rad) [SrvD Ptch_SetPnt parameter]
    float unused_6;      // (6)
    float unused_7;      // (7)
    float unused_8;      // (8)
    float unused_9;      // (9)
    float pitch_angle_actuator_req;      // (10)
    float unused_11;                     // (11)
    float unused_12;                     // (12)
    float unused_13;                     // (13)
    float unused_14;                     // (14)
    float unused_15;                     // (15)
    float unused_16;                     // (16)
    float unused_17;                     // (17)
    float unused_18;                     // (18)
    float unused_19;                     // (19)
    float generator_speed;               // (20) Generator speed (RPM)
    float unused_21;                     // (21)
    float unused_22;                     // (22)
    float unused_23;                     // (23)
    float unused_24;                     // (24)
    float unused_25;                     // (25)
    float unused_26;                     // (26)
    float horizontal_wind_speed;         // (27) Horizontal hub-heigh wind speed (m/s)
    float unused_28;                     // (28)
    float unused_29;                     // (29)
    float unused_30;                     // (30)
    float unused_31;                     // (31)
    float unused_32;                     // (32)
    float pitch_blade2;                  // (33) Blade 2 pitch
    float pitch_blade3;                  // (34) Blade 3 pitch
    float generator_contactor_status;    // (35) Generator contactor status: 1=main (high speed)
                                         // variable-speed generator
    float shaft_brake_status;            // (36) Shaft brake status: 0=off
    float unused_37;                     // (37)
    float unused_38;                     // (38)
    float unused_39;                     // (39)
    float unused_40;                     // (40)
    float demanded_yaw_actuator_torque;  // (41) Demanded yaw actuator torque
    float pitch_command_1;           // (42) Use the command angles of all blades if using individual
                                     // pitch
    float pitch_command_2;           // (43) Use the command angles of all blades if using individual
                                     // pitch
    float pitch_command_3;           // (44) Use the command angles of all blades if using individual
                                     // pitch
    float pitch_command_collective;  // (45) Use the command angle of blade 1 if using collective
                                     // pitch
    float demanded_pitch_rate;       // (46) Demanded pitch rate (Collective pitch)
    float demanded_generator_torque;     // (47) Demanded generator torque
    float demanded_nacelle_yaw_rate;     // (48) Demanded nacelle yaw rate
    float message_array_size;            // (49) avcMSG array size
    float infile_array_size;             // (50) avcINFILE array size
    float outname_array_size;            // (51) avcOUTNAME array size
    float unused_52;                     // (52)
    float unused_53;                     // (53)
    float unused_54;                     // (54)
    float pitch_override;                // (55) Pitch override: 0=yes
    float torque_override;               // (56) Torque override: 0=yes
    float unused_57;                     // (57)
    float unused_58;                     // (58)
    float unused_59;                     // (59)
    float unused_60;                     // (60)
    float n_blades;                      // (61) Number of blades
    float unused_62;                     // (62)
    float unused_63;                     // (63)
    float unused_64;                     // (64)
    float n_log_variables;               // (65) Number of variables returned for logging
    float unused_66;                     // (66)
    float unused_67;                     // (67)
    float unused_68;                     // (68)
    float unused_69;                     // (69)
    float unused_70;                     // (70)
    float unused_71;                     // (71)
    float generator_startup_resistance;  // (72) Generator start-up resistance
    float unused_73;                     // (73)
    float unused_74;                     // (74)
    float unused_75;                     // (75)
    float unused_76;                     // (76)
    float unused_77;                     // (77)
    float unused_78;                     // (78)
    float loads_request;                 // (79) Request for loads: 0=none
    float variable_slip_status;          // (80) Variable slip current status
    float variable_slip_demand;          // (81) Variable slip current demand
};

}  // namespace openturbine::util
