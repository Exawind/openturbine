#include "discon_rotor_test_controller.hpp"

#include <cmath>
#include <cstring>
#include <iostream>

#include "controller_io.hpp"

namespace openturbine::util {

extern "C" {

void PITCH_CONTROLLER(
    float avrSWAP[], int* aviFAIL, const char* const accINFILE, char* const avcOUTNAME,
    char* const avcMSG
) {
    // Map swap from calling program to struct
    ControllerIO io;
    auto swap_array = std::array<float, kSwapArraySize>{};
    std::copy(avrSWAP, std::next(avrSWAP, 81), swap_array.begin());
    io.CopyFromSwapArray(swap_array);

    // Update pitch angle in radians
    io.pitch_command_1 = 2. * M_PI * io.time / 3.;  // Full rotation every 3 seconds
    io.pitch_command_2 = 2. * M_PI * io.time / 6.;  // Full rotation every 6 seconds
    io.pitch_command_3 = 2. * M_PI * io.time / 9.;  // Full rotation every 9 seconds

    // Set failure flag to zero
    *aviFAIL = 0;

    // Set message to success
    strncpy(avcMSG, "success\0", io.message_array_size);
    // *std::next(avcMSG, io.message_array_size - 1) = 0;

    // If this is the first call, output the controller input and output file names
    if (first_call) {
        std::cout << "controller input file: " << accINFILE << std::endl;
        std::cout << "controller output file: " << avcOUTNAME << std::endl;

        // Set first call to false
        first_call = false;
    }

    io.CopyToSwapArray(swap_array);
    std::copy(swap_array.cbegin(), swap_array.cend(), avrSWAP);
}

}  // extern "C"

}  // namespace openturbine::util
