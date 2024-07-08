#include <memory>

#include "discon.h"

namespace openturbine::util {

extern "C" {

bool first_call = true;

void PITCH_CONTROLLER(
    float avrSWAP[], int* aviFAIL, const char* accINFILE, const char* avcOUTNAME, const char* avcMSG
) {
    // Map swap from calling program to struct
    ControllerIO* io = reinterpret_cast<ControllerIO*>(avrSWAP);

    // Update pitch angle in radians (ranges from -90 to 90 starting at zero)
    io->pitch_command_1 = 2. * M_PI * io->time / 3.;
    io->pitch_command_2 = 2. * M_PI * io->time / 6.;
    io->pitch_command_3 = 2. * M_PI * io->time / 9.;

    // Set failure flag to zero
    *aviFAIL = 0;

    // Set message to success
    strncpy(const_cast<char*>(avcMSG), "success\0", static_cast<int>(io->message_array_size));
    const_cast<char*>(avcMSG)[static_cast<int>(io->message_array_size) - 1] = 0;

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
