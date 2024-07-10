#pragma once

#include <functional>
#include <stdexcept>
#include <string>

#include "src/utilities/controllers/controller_io.hpp"
#include "src/vendor/dylib/dylib.hpp"

namespace openturbine::util {

/// A turbine controller class that works as a wrapper around the shared library containing the
/// controller logic
class TurbineController {
public:
    /// Pointer to structure mapping swap array -> named fields i.e. ControllerIO
    ControllerIO* io;

    /// @brief Constructor for the TurbineController class
    /// @param shared_lib_path Path to the shared library containing the controller function
    /// @param controller_function_name Name of the controller function in the shared library
    /// @param input_file_path Path to the input file
    /// @param output_file_path Path to the output file
    TurbineController(
        const std::string& shared_lib_path, const std::string& controller_function_name,
        const std::string& input_file_path = "", const std::string& output_file_path = ""
    );

    // Method to call the controller function from the shared library
    void CallController();

private:
    float swap_array_[kSwapArraySize];  //< Swap array used to pass data to and from the controller
    int status_;                        //< Status of the controller function call
    std::string input_file_path_;       //< Path to the input file
    std::string output_file_path_;      //< Path to the output file
    std::string message_;
    std::string shared_lib_path_;           //< Path to shared library
    std::string controller_function_name_;  //< Name of the controller function in the shared library

    util::dylib lib_;  //< Handle to the shared library
    std::function<void(float*, int*, const char* const, char* const, char* const)>
        controller_function_;  //< Function pointer to the controller function
};

}  // namespace openturbine::util
