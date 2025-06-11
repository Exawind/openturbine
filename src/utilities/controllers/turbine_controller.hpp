#pragma once

#include <functional>
#include <string>

#include "utilities/controllers/controller_io.hpp"
#include "vendor/dylib/dylib.hpp"

namespace openturbine::util {

/// A turbine controller class that works as a wrapper around the shared library containing the
/// controller logic
class TurbineController {
public:
    /// Pointer to structure mapping swap array -> named fields i.e. ControllerIO
    ControllerIO io;

    /// @brief Constructor for the TurbineController class
    /// @param shared_lib_path Path to the shared library containing the controller function
    /// @param controller_function_name Name of the controller function in the shared library
    /// @param input_file_path Path to the input file
    /// @param output_file_path Path to the output file
    TurbineController(
        std::string shared_lib_path, std::string controller_function_name,
        std::string input_file_path, std::string output_file_path
    );

    // Method to call the controller function from the shared library
    void CallController();

private:
    std::string input_file_path_;           //< Path to the input file
    std::string output_file_path_;          //< Path to the output file
    std::string shared_lib_path_;           //< Path to shared library
    std::string controller_function_name_;  //< Name of the controller function in the shared library

    util::dylib lib_;  //< Handle to the shared library
    std::function<void(float*, int*, const char* const, char* const, char* const)>
        controller_function_;  //< Function pointer to the controller function
};

}  // namespace openturbine::util
