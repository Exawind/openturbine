#pragma once

#include <cstring>
#include <stdexcept>
#include <string>

namespace openturbine::util {

// Forward declare the C functions residing in the Aerodyn Inflow shared library to be used in the
// C++ wrapper class
extern "C" {
void ADI_C_PreInit(int*, int*, int*, int*, char*);
void ADI_C_SetupRotor(int*, int*, float*, float*, double*, float*, double*, int*, float*, double*, int*, float*, double*, int*, int*, char*);
void ADI_C_Init(int*, char**, int*, int*, char**, int*, char*, float*, float*, float*, float*, float*, float*, float*, int*, double*, double*, int*, int*, float*, float*, int*, double*, int*, char*, char*, int*, char*);
void ADI_C_SetRotorMotion(int*, float*, double*, float*, float*, float*, double*, float*, float*, float*, double*, float*, float*, int*, float*, double*, float*, float*);
void ADI_C_GetRotorLoads(int*, int*, float*, int*, char*);
void ADI_C_CalcOutput(double*, float*, int*, char*);
void ADI_C_UpdateStates(double*, double*, int*, char*);
void ADI_C_End(int*, char*);
}

/// @brief Wrapper class for the AeroDyn Inflow shared library
class AeroDynInflowLib {
public:
    // Error levels
    enum class ErrorLevel {
        kNone = 0,
        kInfo = 1,
        kWarning = 2,
        kSevereError = 3,
        kFatalError = 4
    };

    // Define some constants
    static constexpr int kErrorMessagesLength{
        1025  // Error message length in Fortran
    };
    static constexpr int kDefaultStringLength{
        1025  // Length of the name used for any output file written by the HD Fortran code
    };

    /// @brief Constructor
    AeroDynInflowLib(const std::string& library_path) : library_path_(library_path), ended_(false) {
        InitializeRoutines();
        InitializeData();
    }

    /// Wrapper for the ADI_C_PreInit routine
    void ADI_PreInit() {
        int n_turbines{1};     // input: Number of turbines
        int transpose_DCM{1};  // input: Transpose the direction cosine matrix?
        int debug_level{0};    // input: Debug level
        int error_status{0};   // output: Error status
        char error_message[kErrorMessagesLength]{'\0'};  // output: Error message buffer

        ADI_C_PreInit(&n_turbines, &transpose_DCM, &debug_level, &error_status, error_message);

        if (error_status != 0) {
            throw std::runtime_error(std::string("PreInit error: ") + error_message);
        }
    }

private:
    std::string library_path_;           // Path to the shared library
    [[maybe_unused]] bool ended_;        // For error handling at end
    [[maybe_unused]] int n_blades_ = 3;  // Default number of blades

    void InitializeRoutines() {
        // Add other routine initializations if needed
    }

    void InitializeData() {
        // Initialize buffers for the class data
        // Similar to the Python __init__ but adapted for C++
    }
};

}  // namespace openturbine::util
