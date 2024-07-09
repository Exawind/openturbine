#include "discon_rotor_test_controller.hpp"

#include "math.h"
#include "stdio.h"
#include "string.h"

#include "controller_io.hpp"

namespace openturbine::util {

extern "C" {

void PITCH_CONTROLLER(
    float avrSWAP[], int* aviFAIL, const char* const accINFILE, char* const avcOUTNAME,
    char* const avcMSG
) {
    // Map swap from calling program to struct
    ControllerIO* io = reinterpret_cast<ControllerIO*>(avrSWAP);

    // Update pitch angle in radians
    io->pitch_command_1 = 2. * M_PI * io->time / 3.;  // Full rotation every 3 seconds
    io->pitch_command_2 = 2. * M_PI * io->time / 6.;  // Full rotation every 6 seconds
    io->pitch_command_3 = 2. * M_PI * io->time / 9.;  // Full rotation every 9 seconds

    // Set failure flag to zero
    *aviFAIL = 0;

    // Set message to success
    strncpy(avcMSG, "success\0", io->message_array_size);
    avcMSG[static_cast<int>(io->message_array_size) - 1] = 0;

    // If this is the first call, output the controller input and output file names
    if (first_call) {
        printf("controller input file: %s\n", accINFILE);
        printf("controller output file: %s\n", avcOUTNAME);

        // Set first call to false
        first_call = false;
    }
}

}  // extern "C"

}  // namespace openturbine::util
