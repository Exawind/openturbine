
#include "discon.h"

// clang-format off

float clamp(float v, float v_min, float v_max)
{
    if (v <= v_min)
    {
        return v_min;
    }
    if (v >= v_max)
    {
        return v_max;
    }
    return v;
}

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
void DISCON(float avrSWAP[], int aviFAIL, char accINFILE[], char avcOUTNAME[], char avcMSG[])
{
    static InternalState state; // Internal state
    static FILE *fp_log = NULL; // Log file pointer
    static FILE *fp_csv = NULL; // CSV file pointer
    float Alpha;                // Current coefficient in the recursive, single-pole, low-pass filter, (-).
    float PitchComI;            // Integral term of command pitch, rad.
    float PitchComP;            // Proportional term of command pitch, rad.
    float PitchComT;            // Total command pitch based on the sum of the proportional and integral terms, rad.

    // Map swap from calling program to struct:
    SwapStruct *swap = (SwapStruct *)avrSWAP;

    // A status flag set by the simulation as follows: 0 if this is the first call, 1 for all subsequent time steps, -1 if this is the final call at the end of the simulation.
    int iStatus = (int)swap->Status;

    // Number of blades, (-).
    int NumBl = (int)swap->NumBl;

    // Initialize aviFAIL to 0
    aviFAIL = 0;

    //--------------------------------------------------------------------------
    // Read External Controller Parameters from the User Interface and init vars
    //--------------------------------------------------------------------------

    // If first call to the DLL
    if (iStatus == 0)
    {
        // Inform users that we are using this user-defined routine:
        aviFAIL = 1;
        strncpy(avcMSG,
                "Running with torque and pitch control of the NREL offshore "
                "5MW baseline wind turbine from DISCON.dll as written by J. "
                "Jonkman of NREL/NWTC for use in the IEA Annex XXIII OC3 "
                "studies.",
                swap->msg_size);

        // Determine some torque control parameters not specified directly:
        state.VS_SySp = VS_RtGnSp / (1.0 + 0.01 * VS_SlPc);
        state.VS_Slope15 = (VS_Rgn2K * VS_Rgn2Sp * VS_Rgn2Sp) / (VS_Rgn2Sp - VS_CtInSp);
        state.VS_Slope25 = (VS_RtPwr / VS_RtGnSp) / (VS_RtGnSp - state.VS_SySp);
        if (VS_Rgn2K == 0.0)
        {
            // Region 2 torque is flat, and thus, the denominator in the else condition is zero
            state.VS_TrGnSp = state.VS_SySp;
        }
        else
        {
            // Region 2 torque is quadratic with speed
            state.VS_TrGnSp = (state.VS_Slope25 - sqrt(state.VS_Slope25 * (state.VS_Slope25 - 4.0 * VS_Rgn2K * state.VS_SySp))) / (2.0 * VS_Rgn2K);
        }

        //----------------------------------------------------------------------
        // Check validity of input parameters
        //----------------------------------------------------------------------

        // Initialize aviFAIL to true (will be set to false if all checks pass)
        aviFAIL = -1;

        if (CornerFreq <= 0.0)
        {
            strncpy(avcMSG, "CornerFreq must be greater than zero.", swap->msg_size);
        }
        else if (VS_DT <= 0.0)
        {
            strncpy(avcMSG, "VS_DT must be greater than zero.", swap->msg_size);
        }
        else if (VS_CtInSp < 0.0)
        {
            strncpy(avcMSG, "VS_CtInSp must not be negative.", swap->msg_size);
        }
        else if (VS_Rgn2Sp <= VS_CtInSp)
        {
            strncpy(avcMSG, "VS_Rgn2Sp must be greater than VS_CtInSp.", swap->msg_size);
        }
        else if (state.VS_TrGnSp < VS_Rgn2Sp)
        {
            strncpy(avcMSG, "VS_TrGnSp must not be less than VS_Rgn2Sp.", swap->msg_size);
        }
        else if (VS_SlPc <= 0.0)
        {
            strncpy(avcMSG, "VS_SlPc must be greater than zero.", swap->msg_size);
        }
        else if (VS_MaxRat <= 0.0)
        {
            strncpy(avcMSG, "VS_MaxRat must be greater than zero.", swap->msg_size);
        }
        else if (VS_RtPwr < 0.0)
        {
            strncpy(avcMSG, "VS_RtPwr must not be negative.", swap->msg_size);
        }
        else if (VS_Rgn2K < 0.0)
        {
            strncpy(avcMSG, "VS_Rgn2K must not be negative.", swap->msg_size);
        }
        else if (VS_Rgn2K * VS_RtGnSp * VS_RtGnSp > VS_RtPwr / VS_RtGnSp)
        {
            strncpy(avcMSG, "VS_Rgn2K*VS_RtGnSp^2 must not be greater than VS_RtPwr/VS_RtGnSp.", swap->msg_size);
        }
        else if (VS_MaxTq < VS_RtPwr / VS_RtGnSp)
        {
            strncpy(avcMSG, "VS_RtPwr/VS_RtGnSp must not be greater than VS_MaxTq.", swap->msg_size);
        }
        else if (PC_DT <= 0.0)
        {
            strncpy(avcMSG, "PC_DT must be greater than zero.", swap->msg_size);
        }
        else if (PC_KI <= 0.0)
        {
            strncpy(avcMSG, "PC_KI must be greater than zero.", swap->msg_size);
        }
        else if (PC_KK <= 0.0)
        {
            strncpy(avcMSG, "PC_KK must be greater than zero.", swap->msg_size);
        }
        else if (PC_RefSpd <= 0.0)
        {
            strncpy(avcMSG, "PC_RefSpd must be greater than zero.", swap->msg_size);
        }
        else if (PC_MaxRat <= 0.0)
        {
            strncpy(avcMSG, "PC_MaxRat must be greater than zero.", swap->msg_size);
        }
        else if (PC_MinPit >= PC_MaxPit)
        {
            strncpy(avcMSG, "PC_MinPit must be less than PC_MaxPit.", swap->msg_size);
        }
        else
        {
            aviFAIL = 0;
            memset(avcMSG, 0, swap->msg_size);
        }

        // If we're debugging the pitch controller, open the debug file and write the header
        if (PC_DbgOut)
        {
            // Allocate memory to store log file paths
            int str_size = swap->outname_size + 7;
            char *file_path = malloc(str_size);

            // Open primary debug file
            snprintf(file_path, str_size, "%s.dbg", avcOUTNAME);
            fp_log = fopen(file_path, "w");

            // Open secondary debug file
            snprintf(file_path, str_size, "%s.dbg2", avcOUTNAME);
            fp_csv = fopen(file_path, "w");

            // Free file path
            free(file_path);

            // Write log header
            fprintf(fp_log, "%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t"
                            "%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\n",
                    "Time", "ElapsedTime", "HorWindV", "GenSpeed", "GenSpeedF", "RelSpdErr",
                    "SpdErr", "IntSpdErr", "GK", "PitchComP", "PitchComI", "PitchComT",
                    "PitchRate1", "PitchRate2", "PitchRate3", "PitchCom1", "PitchCom2", "PitchCom3",
                    "BlPitch1", "BlPitch2", "BlPitch3");
            fprintf(fp_log, "%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t"
                            "%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\n",
                    "(sec)", "(sec)", "(m/sec)", "(rpm)", "(rpm)", "(%)", "(rad/s)",
                    "(rad)", "(-)", "(deg)", "(deg)", "(deg)", "(deg/s)", "(deg/s)",
                    "(deg/s)", "(deg) ", "(deg)", "(deg)", "(deg)", "(deg)", "(deg)");

            // Write CSV header
            fprintf(fp_csv, "%11s", "Time");
            for (int i = 1; i <= 85; i++)
            {
                fprintf(fp_csv, "\tAvrSWAP(%2d)", i);
            }
            fprintf(fp_csv, "\n%11s", "(s)");
            for (int i = 1; i <= 85; i++)
            {
                fprintf(fp_csv, "\t%11s", "(-)");
            }
        }

        // Initialize the state variables
        // NOTE: LastGenTrq, is initialized in the torque controller below for simplicity, not here.
        state.GenSpeedF = swap->GenSpeed;                   // This will ensure that generator speed filter will use the initial value of the generator speed on the first pass
        state.PitchCom[0] = swap->BlPitch1;                 // This will ensure that the variable speed controller picks the correct control region and the pitch controller picks the correct gain on the first call
        state.PitchCom[1] = swap->BlPitch2;                 // ""
        state.PitchCom[2] = swap->BlPitch3;                 // s""
        float GK = 1.0 / (1.0 + state.PitchCom[0] / PC_KK); // This will ensure that the pitch angle is unchanged if the initial SpdErr is zero
        state.IntSpdErr = state.PitchCom[1] / (GK * PC_KI); // This will ensure that the pitch angle is unchanged if the initial SpdErr is zero
        state.LastTime = swap->Time;                        // This will ensure that generator speed filter will use the initial value of the generator speed on the first pass
        state.LastTimePC = swap->Time - PC_DT;              // This will ensure that the pitch  controller is called on the first pass
        state.LastTimeVS = swap->Time - VS_DT;              // This will ensure that the torque controller is called on the first pass
    }

    //--------------------------------------------------------------------------
    // Main control calculations
    //--------------------------------------------------------------------------

    // Only compute control calculations if no error has occurred and we are not on the last time step
    if ((iStatus >= 0) && (aviFAIL >= 0))
    {
        // Abort if the user has not requested a pitch angle actuator (See Appendix A of Bladed User's Guide)
        if ((int)swap->PitchAngleActuatorReq != 0)
        {
            aviFAIL = -1;
            strncpy(avcMSG, "Pitch angle actuator not requested.", swap->msg_size);
        }

        // Set unused outputs to zero (See Appendix A of Bladed User's Guide):
        swap->ShaftBrakeStatus = 0.0;          // Shaft brake status: 0=off
        swap->DemandedYawActuatorTorque = 0.0; // Demanded yaw actuator torque
        swap->DemandedPitchRate = 0.0;         // Demanded pitch rate (Collective pitch)
        swap->DemandedNacelleYawRate = 0.0;    // Demanded nacelle yaw rate
        swap->NumVar = 0.0;                    // Number of variables returned for logging
        swap->GeneratorStartResistance = 0.0;  // Generator start-up resistance
        swap->LoadsReq = 0.0;                  // Request for loads: 0=none
        swap->VariableSlipStatus = 0.0;        // Variable slip current status
        swap->VariableSlipDemand = 0.0;        // Variable slip current demand

        //======================================================================

        // Filter the HSS (generator) speed measurement:
        // NOTE: This is a very simple recursive, single-pole, low-pass filter
        //       with exponential smoothing.

        // Update the coefficient in the recursive formula based on the elapsed
        // time since the last call to the controller:
        Alpha = exp((state.LastTime - swap->Time) * CornerFreq);

        // Apply the filter:
        state.GenSpeedF = (1.0 - Alpha) * swap->GenSpeed + Alpha * state.GenSpeedF;

        //======================================================================

        // Variable-speed torque control:

        // Compute the elapsed time since the last call to the controller:
        float ElapsedTime = swap->Time - state.LastTimeVS;

        // Only perform the control calculations if the elapsed time is greater
        // than or equal to the communication interval of the torque controller:
        // NOTE: Time is scaled by OnePlusEps to ensure that the controller is
        //       called at every time step when VS_DT = DT, even in the presence
        //       of numerical precision errors.

        float GenTrq; // Electrical generator torque, N-m.

        if ((swap->Time * OnePlusEps - state.LastTimeVS) >= VS_DT)
        {
            // Compute the generator torque, which depends on which region we are in:
            if ((state.GenSpeedF >= VS_RtGnSp) || (state.PitchCom[0] >= VS_Rgn3MP))
            {
                // We are in region 3 - power is constant
                GenTrq = VS_RtPwr / state.GenSpeedF;
            }
            else if (state.GenSpeedF <= VS_CtInSp)
            {
                // We are in region 1 - torque is zero
                GenTrq = 0.0;
            }
            else if (state.GenSpeedF < VS_Rgn2Sp)
            {
                // We are in region 1 1/2 - linear ramp in torque from zero to optimal
                GenTrq = state.VS_Slope15 * (state.GenSpeedF - VS_CtInSp);
            }
            else if (state.GenSpeedF < state.VS_TrGnSp)
            {
                // We are in region 2 - optimal torque is proportional to the square of the generator speed
                GenTrq = VS_Rgn2K * state.GenSpeedF * state.GenSpeedF;
            }
            else
            {
                // We are in region 2 1/2 - simple induction generator transition region
                GenTrq = state.VS_Slope25 * (state.GenSpeedF - state.VS_SySp);
            }

            // Saturate the commanded torque using the maximum torque limit
            if (GenTrq > VS_MaxTq)
            {
                GenTrq = VS_MaxTq;
            }

            // Initialize the value of LastGenTrq on the first pass only
            if (iStatus == 0)
            {
                state.LastGenTrq = GenTrq;
            }

            // Torque rate based on the current and last torque commands, N-m/s.
            // Saturate the torque rate using its maximum absolute value
            float TrqRate = clamp((GenTrq - state.LastGenTrq) / ElapsedTime, -VS_MaxRat, VS_MaxRat);

            // Saturate the command using the torque rate limit
            GenTrq = state.LastGenTrq + TrqRate * ElapsedTime;

            // Reset the values of LastTimeVS and LastGenTrq to the current values:
            state.LastTimeVS = swap->Time;
            state.LastGenTrq = GenTrq;
        }

        // Set the generator contactor status, avrSWAP(35), to main (high speed)
        //   variable-speed generator, the torque override to yes, and command the
        //   generator torque (See Appendix A of Bladed User's Guide):

        swap->GeneratorContactorStatus = 1.0;             // Generator contactor status: 1=main (high speed) variable-speed generator
        swap->TorqueOverride = 0.0;                       // Torque override: 0=yes
        swap->DemandedGeneratorTorque = state.LastGenTrq; // Demanded generator torque

        //======================================================================

        // Pitch control:

        // Compute the elapsed time since the last call to the controller:
        ElapsedTime = swap->Time - state.LastTimePC;

        // Only perform the control calculations if the elapsed time is greater than
        //  or equal to the communication interval of the pitch controller:
        // NOTE: Time is scaled by OnePlusEps to ensure that the contoller is called
        //       at every time step when PC_DT = DT, even in the presence of
        //       numerical precision errors.
        if ((swap->Time * OnePlusEps - state.LastTimePC) >= PC_DT)
        {
            // Current value of the gain correction factor, used in the gain
            // scheduling law of the pitch controller, (-).
            // Based on the previously commanded pitch angle for blade 1:
            float GK = 1.0 / (1.0 + state.PitchCom[0] / PC_KK);

            // Compute the current speed error and its integral w.r.t. time; saturate the
            // integral term using the pitch angle limits:
            float SpdErr = state.GenSpeedF - PC_RefSpd;                                                   // Current speed error, rad/s.
            state.IntSpdErr += SpdErr * ElapsedTime;                                                      // Current integral of speed error w.r.t. time
            state.IntSpdErr = clamp(state.IntSpdErr, PC_MinPit / (GK * PC_KI), PC_MaxPit / (GK * PC_KI)); // Saturate the integral term using the pitch angle limits, converted to integral speed error limits

            // Compute the pitch commands associated with the proportional and integral gains:
            PitchComP = GK * PC_KP * SpdErr;          // Proportional term
            PitchComI = GK * PC_KI * state.IntSpdErr; // Integral term (saturated)

            // Superimpose the individual commands to get the total pitch command;
            // saturate the overall command using the pitch angle limits:
            PitchComT = clamp(PitchComP + PitchComI, PC_MinPit, PC_MaxPit);

            // Saturate the overall commanded pitch using the pitch rate limit:
            // NOTE: Since the current pitch angle may be different for each blade
            //       (depending on the type of actuator implemented in the structural
            //       dynamics model), this pitch rate limit calculation and the
            //       resulting overall pitch angle command may be different for each
            //       blade.

            // Current values of the blade pitch angles, rad.
            float BlPitch[3] = {swap->BlPitch1, swap->BlPitch2, swap->BlPitch3};

            // Pitch rates of each blade based on the current pitch angles and current pitch command, rad/s.
            float PitchRate[3];

            // Loop through all blades
            for (int k = 0; k < NumBl; k++)
            {
                PitchRate[k] = (PitchComT - BlPitch[k]) / ElapsedTime;              // Pitch rate of blade K (unsaturated)
                PitchRate[k] = clamp(PitchRate[k], -PC_MaxRat, PC_MaxRat);          // Saturate the pitch rate of blade K using its maximum absolute value
                state.PitchCom[k] = BlPitch[k] + PitchRate[k] * ElapsedTime;        // Saturate the overall command of blade K using the pitch rate limit
                state.PitchCom[k] = clamp(state.PitchCom[k], PC_MinPit, PC_MaxPit); // Saturate the overall command using the pitch angle limits
            }

            // Reset the value of LastTimePC to the current value:
            state.LastTimePC = swap->Time;

            // Output debugging information if requested:
            if (PC_DbgOut)
            {
                fprintf(fp_log, "%11.6f\t%11.4e\t%11.4e\t%11.4e\t%11.4e\t%11.4e\t"
                                "%11.4e\t%11.4e\t%11.4e\t%11.4e\t%11.4e\t%11.4e\t"
                                "%11.4e\t%11.4e\t%11.4e\t%11.4e\t%11.4e\t%11.4e\t"
                                "%11.4e\t%11.4e\t%11.4e\n",
                        swap->Time, ElapsedTime, swap->HorWindV, swap->GenSpeed * RPS2RPM,
                        state.GenSpeedF * RPS2RPM, 100.0 * SpdErr / PC_RefSpd, SpdErr,
                        state.IntSpdErr, GK, PitchComP * R2D, PitchComI * R2D, PitchComT * R2D,
                        PitchRate[0] * R2D, PitchRate[1] * R2D, PitchRate[2] * R2D,
                        state.PitchCom[0] * R2D, state.PitchCom[1] * R2D, state.PitchCom[2] * R2D,
                        BlPitch[0] * R2D, BlPitch[1] * R2D, BlPitch[2] * R2D);
            }
        }

        // Set the pitch override to yes and command the pitch demanded from the last
        // call to the controller (See Appendix A of Bladed User's Guide):
        swap->PitchOverride = 0.0; // Pitch override: 0=yes

        swap->PitchCom1 = state.PitchCom[0]; // Use the command angles of all blades if using individual pitch
        swap->PitchCom2 = state.PitchCom[1]; // "
        swap->PitchCom3 = state.PitchCom[2]; // "

        swap->PitchComCol = state.PitchCom[0]; // Use the command angle of blade 1 if using collective pitch

        if (PC_DbgOut)
        {
            fprintf(fp_csv, "\n%11.6f", swap->Time);
            for (int i = 0; i < 85; i++)
            {
                fprintf(fp_csv, "\t%11.4e", avrSWAP[i]);
            }
        }

        //======================================================================

        // Reset the value of LastTime to the current value:
        state.LastTime = swap->Time;
    }
    else if (iStatus == -8)
    {
        // Pack internal state to file
        FILE *fp = fopen(accINFILE, "wb");
        if (fp)
        {
            fwrite(&state, sizeof(state), 1, fp);
            fclose(fp);
        }
        else
        {
            snprintf(avcMSG, swap->msg_size, "Cannot open file \"%s\". Another program may have locked it for writing", accINFILE);
            aviFAIL = -1;
        }
    }
    else if (iStatus == -9)
    {
        // Unpack internal state from file
        FILE *fp = fopen(accINFILE, "rb");
        if (fp)
        {
            fread(&state, sizeof(state), 1, fp);
            fclose(fp);
        }
        else
        {
            snprintf(avcMSG, swap->msg_size, "Cannot open file \"%s\" for reading. Another program may have locked it.", accINFILE);
            aviFAIL = -1;
        }
    }
}

// clang-format on
