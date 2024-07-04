#pragma once

#include "math.h"
#include "stdbool.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

namespace openturbine::util {

extern "C" {

#define PC_DbgOut 0  // Flag to indicate whether to output debugging information (0=Off)

// Define some constants
// ------------------------------------------------------------------------------------------------
/// Transitional generator speed (HSS side) between regions 1 and 1 1/2, rad/s
static constexpr double kVS_CtInSp{70.16224};
/// Communication interval for torque controller, sec
static constexpr double kVS_DT{0.000125};
/// Maximum torque rate (in absolute value) in torque controller, N-m/s
static constexpr double kVS_MaxRat{15000.};
/// Maximum generator torque in Region 3 (HSS side), N-m
static constexpr double kVS_MaxTq{47402.91};
/// Generator torque constant in Region 2 (HSS side), N-m/(rad/s)^2
static constexpr double kVS_Rgn2K{2.332287};
/// Transitional generator speed (HSS side) between regions 1 1/2 and 2, rad/s
static constexpr double kVS_Rgn2Sp{91.21091};
/// Minimum pitch angle at which the torque is computed as if we are in region 3 regardless of the
/// generator speed, rad
static constexpr double kVS_Rgn3MP{0.01745329};
/// Rated generator speed (HSS side), rad/s
static constexpr double kVS_RtGnSp{121.6805};
/// Rated generator generator power in Region 3, Watts
static constexpr double kVS_RtPwr{5296610.0};
/// Corner frequency (-3dB point) in the recursive, single-pole, low-pass filter, rad/s -- chosen to
/// be 1/4 the blade edgewise natural frequency ( 1/4 of approx. 1 Hz = 0.25 Hz = 1.570796 rad/s)
static constexpr double kCornerFreq{1.570796};
/// A value slightly greater than unity in single precision
static constexpr double kOnePlusEps{1.0 + 1.19e-07};
/// Communication interval for the pitch controller, sec
static constexpr double kPC_DT{0.000125};
/// Integral gain for pitch controller at rated pitch (zero), (-)
static constexpr double kPC_KI{0.008068634};
/// Pitch angle where the the derivative of the aerodynamic power w.r.t. pitch has increased by a
/// factor of two relative to the derivative at rated pitch (zero), rad
static constexpr double kPC_KK{0.1099965};
/// Proportional gain for pitch controller at rated pitch (zero), sec
static constexpr double kPC_KP{0.01882681};
/// Maximum pitch setting in pitch controller, rad
static constexpr double kPC_MaxPit{1.570796};
/// Maximum pitch rate (in absolute value) in pitch controller, rad/s
static constexpr double kPC_MaxRat{0.1396263};
/// Minimum pitch setting in pitch controller, rad
static constexpr double kPC_MinPit{0.0};
/// Desired (reference) HSS speed for pitch controller, rad/s
static constexpr double kPC_RefSpd{122.9096};
/// Factor to convert radians to degrees
static constexpr double kR2D{57.295780};
/// Factor to convert radians per second to revolutions per minute
static constexpr double kRPS2RPM{9.5492966};
/// Rated generator slip percentage in Region 2 1/2, %
static constexpr double kVS_SlPc{10.0};
/// I/O unit for the debugging information
static constexpr int kUnDb{85};
/// I/O unit for the debugging information
static constexpr int kUnDb2{86};
/// I/O unit for pack/unpack (checkpoint & restart)
static constexpr int kUn{87};

/// Structure defining the swap layout (See Appendix A of Bladed User's Guide)
struct SwapStruct {
    float status;                        // (1) Status
    float time;                          // (2) Time
    float unused_3;                      // (3)
    float pitch_blade1;                  // (4) Blade 1 pitch
    float unused_5;                      // (5)
    float unused_6;                      // (6)
    float unused_7;                      // (7)
    float unused_8;                      // (8)
    float unused_9;                      // (9)
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

struct InternalState {
    float generator_speed_filtered;   // Filtered HSS (generator) speed, rad/s.
    float integral_speed_error;       // Current integral of speed error w.r.t. time, rad
    float generator_torque_lastest;   // Commanded electrical generator torque the last time the
                                      // controller was called, N-m
    float time_latest;                // Last time this DLL was called, sec
    float pitch_controller_latest;    // Last time the pitch  controller was called, sec
    float torque_controller_latest;   // Last time the torque controller was called, sec
    float pitch_commanded_latest[3];  // Commanded pitch of each blade the last time the controller
                                      // was called, rad
    float VS_torque_slope_15;         // Torque/speed slope of region 1 1/2 cut-in torque ramp,
                                      // N-m/(rad/s)
    float VS_torque_slope_25;         // Torque/speed slope of region 2 1/2 induction
                                      // generator, N-m/(rad/s)
    float VS_sync_speed;              // Synchronous speed of region 2 1/2 induction generator, rad/s
    float VS_generator_speed_trans;   // Transitional generator speed (HSS side) between regions
                                      // 2 and 2 1/2, rad/s. 1/2, rad/s
};

/// @brief This function is used to clamp a value between a minimum and maximum value
float clamp(float v, float v_min, float v_max) {
    return v < v_min ? v_min : (v > v_max ? v_max : v);
}

/// @brief This Bladed-style DLL controller is used to implement a variable-speed generator-torque
/// controller and PI collective blade pitch controller for the NREL Offshore 5MW baseline wind
/// turbine. This routine was originally written in Fortran by J. Jonkman of NREL/NWTC for use in the
/// IEA Annex XXIII OC3 studies.
/// @param avrSWAP The swap array, used to pass data to, and receive data from, the DLL
///                controller.
/// @param aviFAIL Flag used to indicate the success of this DLL call set as follows:
///                0 if the DLL call was successful,
///                > 0 if the DLL call was successful but cMessage should be issued as a warning
///                messsage,
///                < 0 if the DLL call was unsuccessful or for any other reason the simulation
///                is to be stopped at this point with cMessage as the error message
/// @param accINFILE The name of the parameter input file, 'DISCON.IN'
/// @param avcOUTNAME OUTNAME (Simulation RootName)
/// @param avcMSG MESSAGE (Message from DLL to simulation code [ErrMsg])  The message which will
///               be displayed by the calling program if aviFAIL <> 0
void DISCON(
    float avrSWAP[], int* aviFAIL, const char* accINFILE, const char* avcOUTNAME, const char* avcMSG
);

// Implement a test controller that returns pitch angle in radians (ranges from -90 to 90 starting at
// zero)
void TEST_CONTROLLER(
    float avrSWAP[], int* aviFAIL, const char* accINFILE, const char* avcOUTNAME, const char* avcMSG
);

}  // extern "C"

}  // namespace openturbine::util
