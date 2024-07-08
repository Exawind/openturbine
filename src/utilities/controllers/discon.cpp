#include "discon.h"

#include <memory>

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

// TODO This is a quick and dirty conversion of the DISCON function from the original C code to
// C++. It needs to be refactored to be more idiomatic C++.
void DISCON(
    float avrSWAP[], int* aviFAIL, const char* const accINFILE, char* const avcOUTNAME,
    char* const avcMSG
) {
    // Internal state
    static InternalState state;
    // Current coefficient in the recursive, single-pole, low-pass filter (-)
    float alpha;
    // Integral term of command pitch (rad)
    float pitch_com_integral;
    // Proportional term of command pitch (rad)
    float pitch_com_proportional;
    // Total command pitch based on the sum of the proportional and integral terms (rad)
    float pitch_com_total;
    // Log file pointer
    static FILE* fp_log = nullptr;
    // CSV file pointer
    static FILE* fp_csv = nullptr;

    // Map swap from calling program to struct
    ControllerIO* swap = reinterpret_cast<ControllerIO*>(avrSWAP);

    // A status flag set by the simulation as follows: 0 if this is the first call, 1 for all
    // subsequent time steps, -1 if this is the final call at the end of the simulation
    int status = static_cast<int>(swap->status);

    // Number of blades, (-)
    int num_blades = static_cast<int>(swap->n_blades);

    // Initialize aviFAIL to 0
    *aviFAIL = 0;

    //--------------------------------------------------------------------------
    // Read External Controller Parameters from the User Interface and initialize variables
    //--------------------------------------------------------------------------

    // If first call to the DLL
    if (status == 0) {
        // Inform users that we are using this user-defined routine:
        *aviFAIL = 1;
        strncpy(
            const_cast<char*>(avcMSG),
            "Running with torque and pitch control of the NREL offshore "
            "5MW baseline wind turbine from DISCON.dll as written by J. "
            "Jonkman of NREL/NWTC for use in the IEA Annex XXIII OC3 "
            "studies.",
            swap->message_array_size
        );

        // Determine some torque control parameters not specified directly
        state.VS_sync_speed = kVS_RtGnSp / (1. + 0.01 * kVS_SlPc);
        state.VS_torque_slope_15 = (kVS_Rgn2K * kVS_Rgn2Sp * kVS_Rgn2Sp) / (kVS_Rgn2Sp - kVS_CtInSp);
        state.VS_torque_slope_25 = (kVS_RtPwr / kVS_RtGnSp) / (kVS_RtGnSp - state.VS_sync_speed);
        if (kVS_Rgn2K == 0.) {
            // Region 2 torque is flat, and thus, the denominator in the else condition is zero
            state.VS_generator_speed_trans = state.VS_sync_speed;
        } else {
            // Region 2 torque is quadratic with speed
            state.VS_generator_speed_trans =
                (state.VS_torque_slope_25 -
                 sqrt(
                     state.VS_torque_slope_25 *
                     (state.VS_torque_slope_25 - 4.0 * kVS_Rgn2K * state.VS_sync_speed)
                 )) /
                (2.0 * kVS_Rgn2K);
        }

        //----------------------------------------------------------------------
        // Check validity of input parameters
        //----------------------------------------------------------------------

        // Initialize aviFAIL to true (will be set to false if all checks pass)
        *aviFAIL = -1;

        if (kCornerFreq <= 0.0) {
            strncpy(
                const_cast<char*>(avcMSG), "CornerFreq must be greater than zero.",
                swap->message_array_size
            );
        } else if (kVS_DT <= 0.0) {
            strncpy(
                const_cast<char*>(avcMSG), "VS_DT must be greater than zero.",
                swap->message_array_size
            );
        } else if (kVS_CtInSp < 0.0) {
            strncpy(
                const_cast<char*>(avcMSG), "VS_CtInSp must not be negative.",
                swap->message_array_size
            );
        } else if (kVS_Rgn2Sp <= kVS_CtInSp) {
            strncpy(
                const_cast<char*>(avcMSG), "VS_Rgn2Sp must be greater than VS_CtInSp.",
                swap->message_array_size
            );
        } else if (state.VS_generator_speed_trans < kVS_Rgn2Sp) {
            strncpy(
                const_cast<char*>(avcMSG),
                "VS_generator_speed_trans must not be less than VS_Rgn2Sp.", swap->message_array_size
            );
        } else if (kVS_SlPc <= 0.0) {
            strncpy(
                const_cast<char*>(avcMSG), "VS_SlPc must be greater than zero.",
                swap->message_array_size
            );
        } else if (kVS_MaxRat <= 0.0) {
            strncpy(
                const_cast<char*>(avcMSG), "VS_MaxRat must be greater than zero.",
                swap->message_array_size
            );
        } else if (kVS_RtPwr < 0.0) {
            strncpy(
                const_cast<char*>(avcMSG), "VS_RtPwr must not be negative.", swap->message_array_size
            );
        } else if (kVS_Rgn2K < 0.0) {
            strncpy(
                const_cast<char*>(avcMSG), "VS_Rgn2K must not be negative.", swap->message_array_size
            );
        } else if (kVS_Rgn2K * kVS_RtGnSp * kVS_RtGnSp > kVS_RtPwr / kVS_RtGnSp) {
            strncpy(
                const_cast<char*>(avcMSG),
                "VS_Rgn2K*VS_RtGnSp^2 must not be greater than VS_RtPwr/VS_RtGnSp.",
                swap->message_array_size
            );
        } else if (kVS_MaxTq < kVS_RtPwr / kVS_RtGnSp) {
            strncpy(
                const_cast<char*>(avcMSG), "VS_RtPwr/VS_RtGnSp must not be greater than VS_MaxTq.",
                swap->message_array_size
            );
        } else if (kPC_DT <= 0.0) {
            strncpy(
                const_cast<char*>(avcMSG), "PC_DT must be greater than zero.",
                swap->message_array_size
            );
        } else if (kPC_KI <= 0.0) {
            strncpy(
                const_cast<char*>(avcMSG), "PC_KI must be greater than zero.",
                swap->message_array_size
            );
        } else if (kPC_KK <= 0.0) {
            strncpy(
                const_cast<char*>(avcMSG), "PC_KK must be greater than zero.",
                swap->message_array_size
            );
        } else if (kPC_RefSpd <= 0.0) {
            strncpy(
                const_cast<char*>(avcMSG), "PC_RefSpd must be greater than zero.",
                swap->message_array_size
            );
        } else if (kPC_MaxRat <= 0.0) {
            strncpy(
                const_cast<char*>(avcMSG), "PC_MaxRat must be greater than zero.",
                swap->message_array_size
            );
        } else if (kPC_MinPit >= kPC_MaxPit) {
            strncpy(
                const_cast<char*>(avcMSG), "PC_MinPit must be less than PC_MaxPit.",
                swap->message_array_size
            );
        } else {
            *aviFAIL = 0;
            memset(const_cast<char*>(avcMSG), 0, swap->message_array_size);
        }

        // If we're debugging the pitch controller, open the debug file and write the header
        if (PC_DbgOut) {
            // Allocate memory to store log file paths
            int str_size = swap->outname_array_size + 7;
            std::unique_ptr<char[]> file_path(new char[str_size]);

            // Open primary debug file
            snprintf(file_path.get(), str_size, "%s.dbg", avcOUTNAME);
            fp_log = fopen(file_path.get(), "w");

            // Open secondary debug file
            snprintf(file_path.get(), str_size, "%s.dbg2", avcOUTNAME);
            fp_csv = fopen(file_path.get(), "w");

            // Write log header
            fprintf(
                fp_log,
                "%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t"
                "%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\n",
                "Time", "ElapsedTime", "horizontal_wind_speed", "generator_speed",
                "generator_speed_filtered", "RelSpdErr", "SpdErr", "integral_speed_error", "GK",
                "pitch_commanded_latestP", "pitch_commanded_latestI", "pitch_commanded_latestT",
                "PitchRate1", "PitchRate2", "PitchRate3", "pitch_command_1", "pitch_command_2",
                "pitch_command_3", "pitch_blade1", "pitch_blade2", "pitch_blade3"
            );
            fprintf(
                fp_log,
                "%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t"
                "%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\t%11s\n",
                "(sec)", "(sec)", "(m/sec)", "(rpm)", "(rpm)", "(%)", "(rad/s)", "(rad)", "(-)",
                "(deg)", "(deg)", "(deg)", "(deg/s)", "(deg/s)", "(deg/s)", "(deg) ", "(deg)",
                "(deg)", "(deg)", "(deg)", "(deg)"
            );

            // Write CSV header
            fprintf(fp_csv, "%11s", "Time");
            for (int i = 1; i <= 85; i++) {
                fprintf(fp_csv, "\tAvrSWAP(%2d)", i);
            }
            fprintf(fp_csv, "\n%11s", "(s)");
            for (int i = 1; i <= 85; i++) {
                fprintf(fp_csv, "\t%11s", "(-)");
            }
        }

        // Initialize the state variables
        // NOTE: generator_torque_lastest is initialized in the torque controller below for
        // simplicity (not here).
        // --------------------------------------------------------------------------------------------
        // This will ensure that generator speed filter will use the initial value of the
        // generator speed on the first pass
        state.generator_speed_filtered = swap->generator_speed;

        // This will ensure that the variable speed controller picks the correct control region
        // and the pitch controller picks the correct gain on the first call
        state.pitch_commanded_latest[0] = swap->pitch_blade1;
        state.pitch_commanded_latest[1] = swap->pitch_blade2;
        state.pitch_commanded_latest[2] = swap->pitch_blade3;

        // This will ensure that the pitch angle is unchanged if the initial SpdErr is zero
        float GK = 1.0 / (1.0 + state.pitch_commanded_latest[0] / kPC_KK);

        // This will ensure that the pitch angle is unchanged if the initial SpdErr is zero
        state.integral_speed_error = state.pitch_commanded_latest[1] / (GK * kPC_KI);

        // This will ensure that generator speed filter will use the initial value of the
        // generator speed on the first pass
        state.time_latest = swap->time;

        // This will ensure that the pitch controller is called on the first pass
        state.pitch_controller_latest = swap->time - kPC_DT;

        // This will ensure that the torque controller is called on the first pass
        state.torque_controller_latest = swap->time - kVS_DT;
    }

    //--------------------------------------------------------------------------
    // Main control calculations
    //--------------------------------------------------------------------------

    // Only compute control calculations if no error has occurred and we are not on the last time
    // step
    if (status >= 0 && *aviFAIL >= 0) {
        // Abort if the user has not requested a pitch angle actuator (See Appendix A of Bladed
        // User's Guide)
        if (static_cast<int>(swap->pitch_angle_actuator_req) != 0) {
            *aviFAIL = -1;
            strncpy(
                const_cast<char*>(avcMSG), "Pitch angle actuator not requested.",
                swap->message_array_size
            );
        }

        // Set unused outputs to zero (See Appendix A of Bladed User's Guide):
        swap->shaft_brake_status = 0.;            // Shaft brake status: 0=off
        swap->demanded_yaw_actuator_torque = 0.;  // Demanded yaw actuator torque
        swap->demanded_pitch_rate = 0.;           // Demanded pitch rate (Collective pitch)
        swap->demanded_nacelle_yaw_rate = 0.;     // Demanded nacelle yaw rate
        swap->n_log_variables = 0.;               // Number of variables returned for logging
        swap->generator_startup_resistance = 0.;  // Generator start-up resistance
        swap->loads_request = 0.;                 // Request for loads: 0=none
        swap->variable_slip_status = 0.;          // Variable slip current status
        swap->variable_slip_demand = 0.;          // Variable slip current demand

        //======================================================================
        // Filter the HSS (generator) speed measurement
        // NOTE: This is a very simple recursive, single-pole, low-pass filter with exponential
        // smoothing

        // Update the coefficient in the recursive formula based on the elapsed time since the
        // last call to the controller
        alpha = exp((state.time_latest - swap->time) * kCornerFreq);

        // Apply the filter
        state.generator_speed_filtered =
            (1. - alpha) * swap->generator_speed + alpha * state.generator_speed_filtered;

        // ==========================================================================
        // Variable-speed torque control

        // Compute the elapsed time since the last call to the controller
        float elapsed_time = swap->time - state.torque_controller_latest;

        // Only perform the control calculations if the elapsed time is greater than or equal to
        // the communication interval of the torque controller NOTE: Time is scaled by OnePlusEps
        // to ensure that the controller is called at every time step when kVS_DT = DT, even in
        // the presence of numerical precision errors

        float gen_trq;  // Electrical generator torque, N-m

        if ((swap->time * kOnePlusEps - state.torque_controller_latest) >= kVS_DT) {
            // Compute the generator torque, which depends on which region we are in
            if ((state.generator_speed_filtered >= kVS_RtGnSp) ||
                (state.pitch_commanded_latest[0] >= kVS_Rgn3MP)) {
                // We are in region 3 - power is constant
                gen_trq = kVS_RtPwr / state.generator_speed_filtered;
            } else if (state.generator_speed_filtered <= kVS_CtInSp) {
                // We are in region 1 - torque is zero
                gen_trq = 0.0;
            } else if (state.generator_speed_filtered < kVS_Rgn2Sp) {
                // We are in region 1 1/2 - linear ramp in torque from zero to optimal
                gen_trq = state.VS_torque_slope_15 * (state.generator_speed_filtered - kVS_CtInSp);
            } else if (state.generator_speed_filtered < state.VS_generator_speed_trans) {
                // We are in region 2 - optimal torque is proportional to the square of the
                // generator speed
                gen_trq =
                    kVS_Rgn2K * state.generator_speed_filtered * state.generator_speed_filtered;
            } else {
                // We are in region 2 1/2 - simple induction generator transition region
                gen_trq = state.VS_torque_slope_25 *
                          (state.generator_speed_filtered - state.VS_sync_speed);
            }

            // Saturate the commanded torque using the maximum torque limit
            if (gen_trq > kVS_MaxTq) {
                gen_trq = kVS_MaxTq;
            }

            // Initialize the value of generator_torque_lastest on the first pass only
            if (status == 0) {
                state.generator_torque_lastest = gen_trq;
            }

            // Torque rate based on the current and last torque commands, N-m/s
            // Saturate the torque rate using its maximum absolute value
            float trq_rate = clamp(
                (gen_trq - state.generator_torque_lastest) / elapsed_time, -kVS_MaxRat, kVS_MaxRat
            );

            // Saturate the command using the torque rate limit
            gen_trq = state.generator_torque_lastest + trq_rate * elapsed_time;

            // Reset the values of torque_controller_latest and generator_torque_lastest to the
            // current values
            state.torque_controller_latest = swap->time;
            state.generator_torque_lastest = gen_trq;
        }

        // Set the generator contactor status, avrSWAP(35), to main (high speed)
        //   variable-speed generator, the torque override to yes, and command the
        //   generator torque (See Appendix A of Bladed User's Guide):

        swap->generator_contactor_status =
            1.0;  // Generator contactor status: 1=main (high speed) variable-speed generator
        swap->torque_override = 0.0;  // Torque override: 0=yes
        swap->demanded_generator_torque =
            state.generator_torque_lastest;  // Demanded generator torque

        //======================================================================

        // Pitch control:

        // Compute the elapsed time since the last call to the controller:
        elapsed_time = swap->time - state.pitch_controller_latest;

        // Only perform the control calculations if the elapsed time is greater than or equal to
        // the communication interval of the pitch controller NOTE: Time is scaled by OnePlusEps
        // to ensure that the contoller is called at every time step when PC_DT = DT, even in the
        // presence of numerical precision errors
        if ((swap->time * kOnePlusEps - state.pitch_controller_latest) >= kPC_DT) {
            // Current value of the gain correction factor, used in the gain
            // scheduling law of the pitch controller, (-).
            // Based on the previously commanded pitch angle for blade 1:
            float GK = 1.0 / (1.0 + state.pitch_commanded_latest[0] / kPC_KK);

            // Compute the current speed error and its integral w.r.t. time; saturate the
            // integral term using the pitch angle limits
            float speed_error = state.generator_speed_filtered - kPC_RefSpd;
            state.integral_speed_error += speed_error * elapsed_time;
            state.integral_speed_error = clamp(
                state.integral_speed_error, kPC_MinPit / (kOnePlusEps * kPC_KI),
                kPC_MaxPit / (kOnePlusEps * kPC_KI)
            );

            // Compute the pitch commands associated with the proportional and integral gains
            pitch_com_proportional = GK * kPC_KP * speed_error;
            pitch_com_integral = GK * kPC_KI * state.integral_speed_error;

            // Superimpose the individual commands to get the total pitch command; saturate the
            // overall command using the pitch angle limits
            pitch_com_total =
                clamp(pitch_com_proportional + pitch_com_integral, kPC_MinPit, kPC_MaxPit);

            // Saturate the overall commanded pitch using the pitch rate limit:
            // NOTE: Since the current pitch angle may be different for each blade
            //       (depending on the type of actuator implemented in the structural
            //       dynamics model), this pitch rate limit calculation and the
            //       resulting overall pitch angle command may be different for each
            //       blade.

            // Current values of the blade pitch angles, rad
            float blade_pitch[3] = {swap->pitch_blade1, swap->pitch_blade2, swap->pitch_blade3};

            // Pitch rates of each blade based on the current pitch angles and current pitch
            // command, rad/s
            float pitch_rate[3];

            // Loop through all blades
            for (int k = 0; k < num_blades; k++) {
                // Pitch rate of blade K (unsaturated)
                pitch_rate[k] = (pitch_com_total - blade_pitch[k]) / elapsed_time;
                // Saturate the pitch rate of blade K using its maximum absolute value
                pitch_rate[k] = clamp(pitch_rate[k], -kPC_MaxRat, kPC_MaxRat);
                // Saturate the overall command of blade K using the pitch rate limit
                state.pitch_commanded_latest[k] = blade_pitch[k] + pitch_rate[k] * elapsed_time;
                // Saturate the overall command using the pitch angle limits
                state.pitch_commanded_latest[k] =
                    clamp(state.pitch_commanded_latest[k], kPC_MinPit, kPC_MaxPit);
            }

            // Reset the value of pitch_controller_latest to the current value
            state.pitch_controller_latest = swap->time;

            // Output debugging information if requested:
            if (PC_DbgOut) {
                fprintf(
                    fp_log,
                    "%11.6f\t%11.4e\t%11.4e\t%11.4e\t%11.4e\t%11.4e\t"
                    "%11.4e\t%11.4e\t%11.4e\t%11.4e\t%11.4e\t%11.4e\t"
                    "%11.4e\t%11.4e\t%11.4e\t%11.4e\t%11.4e\t%11.4e\t"
                    "%11.4e\t%11.4e\t%11.4e\n",
                    swap->time, elapsed_time, swap->horizontal_wind_speed,
                    swap->generator_speed * kRPS2RPM, state.generator_speed_filtered * kRPS2RPM,
                    100.0 * speed_error / kPC_RefSpd, speed_error, state.integral_speed_error, GK,
                    pitch_com_proportional * kR2D, pitch_com_integral * kR2D, pitch_com_total * kR2D,
                    pitch_rate[0] * kR2D, pitch_rate[1] * kR2D, pitch_rate[2] * kR2D,
                    state.pitch_commanded_latest[0] * kR2D, state.pitch_commanded_latest[1] * kR2D,
                    state.pitch_commanded_latest[2] * kR2D, blade_pitch[0] * kR2D,
                    blade_pitch[1] * kR2D, blade_pitch[2] * kR2D
                );
            }
        }

        // Set the pitch override to yes and command the pitch demanded from the last
        // call to the controller (See Appendix A of Bladed User's Guide):
        swap->pitch_override = 0.;  // Pitch override: 0=yes

        swap->pitch_command_1 = state.pitch_commanded_latest[0];  // Use the command angles of all
                                                                  // blades if using individual pitch
        swap->pitch_command_2 = state.pitch_commanded_latest[1];  // "
        swap->pitch_command_3 = state.pitch_commanded_latest[2];  // "

        swap->pitch_command_collective =
            state.pitch_commanded_latest[0];  // Use the command angle of blade 1 if using collective
                                              // pitch

        if (PC_DbgOut) {
            fprintf(fp_csv, "\n%11.6f", swap->time);
            for (int i = 0; i < 85; i++) {
                fprintf(fp_csv, "\t%11.4e", avrSWAP[i]);
            }
        }

        //======================================================================

        // Reset the value of time_latest to the current value:
        state.time_latest = swap->time;
    } else if (status == -8) {
        // Pack internal state to file
        FILE* fp = fopen(accINFILE, "wb");
        if (fp) {
            [[maybe_unused]] auto f = fwrite(&state, sizeof(state), 1, fp);
            fclose(fp);
        } else {
            snprintf(
                const_cast<char*>(avcMSG), swap->message_array_size,
                "Cannot open file \"%s\". Another program may have locked it for writing", accINFILE
            );
            *aviFAIL = -1;
        }
    } else if (status == -9) {
        // Unpack internal state from file
        FILE* fp = fopen(accINFILE, "rb");
        if (fp) {
            [[maybe_unused]] auto f = fread(&state, sizeof(state), 1, fp);
            fclose(fp);
        } else {
            snprintf(
                const_cast<char*>(avcMSG), swap->message_array_size,
                "Cannot open file \"%s\" for reading. Another program may have locked it.", accINFILE
            );
            *aviFAIL = -1;
        }
    }
}

}  // extern "C"

}  // namespace openturbine::util
