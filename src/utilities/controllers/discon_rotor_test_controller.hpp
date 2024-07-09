#pragma once

namespace openturbine::util {

extern "C" {

auto first_call = true;

/// Implement a test controller that returns pitch angle in radians (ranges from -90 to 90 starting
/// at zero) - used for testing purposes
void PITCH_CONTROLLER(
    float avrSWAP[], int* aviFAIL, const char* const accINFILE, char* const avcOUTNAME,
    char* const avcMSG
);

}  // extern "C"

}  // namespace openturbine::util
