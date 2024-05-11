//**********************************************************************************************************************************
// LICENSING
// Copyright (C) 2015-2016  National Renewable Energy Laboratory
// Copyright (C) 2016-2017  Envision Energy USA, LTD
// Copyright (C) 2023       National Renewable Energy Laboratory
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//**********************************************************************************************************************************

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// clang-format off

#include "stdbool.h"
#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "math.h"

#define PC_DbgOut 1 // Flag to indicate whether to output debugging information (0=Off)

// Constants
const float VS_CtInSp = 70.16224;        // Transitional generator speed (HSS side) between regions 1 and 1 1/2, rad/s.
const float VS_DT = 0.000125;            // JASON:THIS CHANGED FOR ITI BARGE:      0.0001   // Communication interval for torque controller, sec.
const float VS_MaxRat = 15000.0;         // Maximum torque rate (in absolute value) in torque controller, N-m/s.
const float VS_MaxTq = 47402.91;         // Maximum generator torque in Region 3 (HSS side), N-m. -- chosen to be 10% above VS_RtTq 43.09355kNm
const float VS_Rgn2K = 2.332287;         // Generator torque constant in Region 2 (HSS side), N-m/(rad/s)^2.
const float VS_Rgn2Sp = 91.21091;        // Transitional generator speed (HSS side) between regions 1 1/2 and 2, rad/s.
const float VS_Rgn3MP = 0.01745329;      // Minimum pitch angle at which the torque is computed as if we are in region 3 regardless of the generator speed, rad. -- chosen to be 1.0 degree above PC_MinPit
const float VS_RtGnSp = 121.6805;        // Rated generator speed (HSS side), rad/s. -- chosen to be 99% of PC_RefSpd
const float VS_RtPwr = 5296610.0;        // Rated generator generator power in Region 3, Watts. -- chosen to be 5MW divided by the electrical generator efficiency of 94.4%
const float CornerFreq = 1.570796;       // Corner frequency (-3dB point) in the recursive, single-pole, low-pass filter, rad/s. -- chosen to be 1/4 the blade edgewise natural frequency ( 1/4 of approx. 1Hz = 0.25Hz = 1.570796rad/s)
const float OnePlusEps = 1.0 + 1.19e-07; // The number slightly greater than unity in single precision.
const float PC_DT = 0.000125;            // JASON:THIS CHANGED FOR ITI BARGE:      0.0001                    // Communication interval for pitch  controller, sec.
const float PC_KI = 0.008068634;         // Integral gain for pitch controller at rated pitch (zero), (-).
const float PC_KK = 0.1099965;           // Pitch angle where the the derivative of the aerodynamic power w.r.t. pitch has increased by a factor of two relative to the derivative at rated pitch (zero), rad.
const float PC_KP = 0.01882681;          // Proportional gain for pitch controller at rated pitch (zero), sec.
const float PC_MaxPit = 1.570796;        // Maximum pitch setting in pitch controller, rad.
const float PC_MaxRat = 0.1396263;       // Maximum pitch  rate (in absolute value) in pitch  controller, rad/s.
const float PC_MinPit = 0.0;             // Minimum pitch setting in pitch controller, rad.
const float PC_RefSpd = 122.9096;        // Desired (reference) HSS speed for pitch controller, rad/s.
const float R2D = 57.295780;             // Factor to convert radians to degrees.
const float RPS2RPM = 9.5492966;         // Factor to convert radians per second to revolutions per minute.
const float VS_SlPc = 10.0;              // Rated generator slip percentage in Region 2 1/2, %.
const int UnDb = 85;                     // I/O unit for the debugging information
const int UnDb2 = 86;                    // I/O unit for the debugging information
const int Un = 87;                       // I/O unit for pack/unpack (checkpoint & restart)

// Structure defining swap layout (See Appendix A of Bladed User's Guide)
typedef struct SwapStruct
{
    float Status;                    // (1) Status
    float Time;                      // (2) Time
    float unused_3;                  // (3)
    float BlPitch1;                  // (4) Blade 1 pitch
    float unused_5;                  // (5)
    float unused_6;                  // (6)
    float unused_7;                  // (7)
    float unused_8;                  // (8)
    float unused_9;                  // (9)
    float PitchAngleActuatorReq;     // (10)
    float unused_11;                 // (11)
    float unused_12;                 // (12)
    float unused_13;                 // (13)
    float unused_14;                 // (14)
    float unused_15;                 // (15)
    float unused_16;                 // (16)
    float unused_17;                 // (17)
    float unused_18;                 // (18)
    float unused_19;                 // (19)
    float GenSpeed;                  // (20) Generator speed (RPM)
    float unused_21;                 // (21)
    float unused_22;                 // (22)
    float unused_23;                 // (23)
    float unused_24;                 // (24)
    float unused_25;                 // (25)
    float unused_26;                 // (26)
    float HorWindV;                  // (27) Horizontal hub-heigh wind speed (m/s)
    float unused_28;                 // (28)
    float unused_29;                 // (29)
    float unused_30;                 // (30)
    float unused_31;                 // (31)
    float unused_32;                 // (32)
    float BlPitch2;                  // (33) Blade 2 pitch
    float BlPitch3;                  // (34) Blade 3 pitch
    float GeneratorContactorStatus;  // (35) Generator contactor status: 1=main (high speed) variable-speed generator
    float ShaftBrakeStatus;          // (36) Shaft brake status: 0=off
    float unused_37;                 // (37)
    float unused_38;                 // (38)
    float unused_39;                 // (39)
    float unused_40;                 // (40)
    float DemandedYawActuatorTorque; // (41) Demanded yaw actuator torque
    float PitchCom1;                 // (42) Use the command angles of all blades if using individual pitch
    float PitchCom2;                 // (43) Use the command angles of all blades if using individual pitch
    float PitchCom3;                 // (44) Use the command angles of all blades if using individual pitch
    float PitchComCol;               // (45) Use the command angle of blade 1 if using collective pitch
    float DemandedPitchRate;         // (46) Demanded pitch rate (Collective pitch)
    float DemandedGeneratorTorque;   // (47) Demanded generator torque
    float DemandedNacelleYawRate;    // (48) Demanded nacelle yaw rate
    float msg_size;                  // (49) avcMSG array size
    float infile_size;               // (50) avcINFILE array size
    float outname_size;              // (51) avcOUTNAME array size
    float unused_52;                 // (52)
    float unused_53;                 // (53)
    float unused_54;                 // (54)
    float PitchOverride;             // (55) Pitch override: 0=yes
    float TorqueOverride;            // (56) Torque override: 0=yes
    float unused_57;                 // (57)
    float unused_58;                 // (58)
    float unused_59;                 // (59)
    float unused_60;                 // (60)
    float NumBl;                     // (61) Number of blades
    float unused_62;                 // (62)
    float unused_63;                 // (63)
    float unused_64;                 // (64)
    float NumVar;                    // (65) Number of variables returned for logging
    float unused_66;                 // (66)
    float unused_67;                 // (67)
    float unused_68;                 // (68)
    float unused_69;                 // (69)
    float unused_70;                 // (70)
    float unused_71;                 // (71)
    float GeneratorStartResistance;  // (72) Generator start-up resistance
    float unused_73;                 // (73)
    float unused_74;                 // (74)
    float unused_75;                 // (75)
    float unused_76;                 // (76)
    float unused_77;                 // (77)
    float unused_78;                 // (78)
    float LoadsReq;                  // (79) Request for loads: 0=none
    float VariableSlipStatus;        // (80) Variable slip current status
    float VariableSlipDemand;        // (81) Variable slip current demand
} SwapStruct;

typedef struct InternalState
{
    float GenSpeedF;   // Filtered HSS (generator) speed, rad/s.
    float IntSpdErr;   // Current integral of speed error w.r.t. time, rad.
    float LastGenTrq;  // Commanded electrical generator torque the last time the controller was called, N-m.
    float LastTime;    // Last time this DLL was called, sec.
    float LastTimePC;  // Last time the pitch  controller was called, sec.
    float LastTimeVS;  // Last time the torque controller was called, sec.
    float PitchCom[3]; // Commanded pitch of each blade the last time the controller was called, rad.
    float VS_Slope15;  // Torque/speed slope of region 1 1/2 cut-in torque ramp , N-m/(rad/s).
    float VS_Slope25;  // Torque/speed slope of region 2 1/2 induction generator, N-m/(rad/s).
    float VS_SySp;     // Synchronous speed of region 2 1/2 induction generator, rad/s.
    float VS_TrGnSp;   // Transitional generator speed (HSS side) between regions 2 and 2 1/2, rad/s.
} InternalState;

float clamp(float v, float v_min, float v_max);

//------------------------------------------------------------------------------
// This Bladed-style DLL controller is used to implement a variable-speed
// generator-torque controller and PI collective blade pitch controller for
// the NREL Offshore 5MW baseline wind turbine.  This routine was written by
// J. Jonkman of NREL/NWTC for use in the IEA Annex XXIII OC3 studies.
//
// avrSWAP     - The swap array, used to pass data to, and receive data from, the DLL controller.
// aviFAIL     - Flag used to indicate the success of this DLL call set as follows: 0 if the DLL call was successful, >0 if the DLL call was successful but cMessage should be issued as a warning messsage, <0 if the DLL call was unsuccessful or for any other reason the simulation is to be stopped at this point with cMessage as the error message.
// accINFILE   - The name of the parameter input file, 'DISCON.IN'.
// avcOUTNAME  - OUTNAME (Simulation RootName)
// avcMSG      - MESSAGE (Message from DLL to simulation code [ErrMsg])  The message which will be displayed by the calling program if aviFAIL <> 0.
//------------------------------------------------------------------------------
void DISCON(float avrSWAP[], int aviFAIL, char accINFILE[], char avcOUTNAME[], char avcMSG[]);

// clang-format on

#ifdef __cplusplus
}
#endif
