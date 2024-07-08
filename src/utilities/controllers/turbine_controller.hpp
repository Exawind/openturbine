#pragma once

#include <functional>
#include <stdexcept>
#include <string>

#include "src/utilities/controllers/discon.h"
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
        std::string shared_lib_path, std::string controller_function_name,
        std::string input_file_path = "", std::string output_file_path = ""
    )
        : input_file_path_(input_file_path),
          output_file_path_(output_file_path),
          shared_lib_path_(shared_lib_path),
          controller_function_name_(controller_function_name),
          lib_(shared_lib_path, util::dylib::no_filename_decorations) {
        // Make sure we have a valid shared library path + controller function name
        try {
            this->controller_function_ = lib_.get_function<void(
                float* avrSWAP, int* aviFAIL, const char* accINFILE, const char* avcOUTNAME,
                const char* avcMSG
            )>(this->controller_function_name_);
        } catch (const util::dylib::load_error& e) {
            throw std::runtime_error("Failed to load shared library: " + shared_lib_path);
        } catch (const util::dylib::symbol_error& e) {
            throw std::runtime_error("Failed to get function: " + controller_function_name);
        }

        // Initialize some values required for calling the controller function
        for (int i = 0; i < 81; ++i) {
            // Initialize swap array to zero
            this->swap_array_[i] = 0.;
        }
        this->status_ = 0;                        // Status of the controller function call
        this->message_ = std::string(1024, ' ');  // 1024 characters for message

        // Map swap array to ControllerIO structure for easier access
        this->io = reinterpret_cast<ControllerIO*>(this->swap_array_);
        this->io->infile_array_size = input_file_path.size();
        this->io->outname_array_size = output_file_path.size();
        this->io->message_array_size = this->message_.size();
    }

    // Method to call the controller function from the shared library
    void CallController() {
        this->controller_function_(
            this->swap_array_, &this->status_, this->input_file_path_.data(),
            this->output_file_path_.data(), this->message_.data()
        );

        if (this->status_ < 0) {
            throw std::runtime_error("Error raised in controller: " + this->message_);
        } else if (this->status_ > 0) {
            std::cout << "Warning from controller: " << this->message_ << std::endl;
        }
    }

private:
    float swap_array_[81];          //< Swap array used to pass data to and from the controller
    int status_;                    //< Status of the controller function call
    std::string input_file_path_;   //< Path to the input file
    std::string output_file_path_;  //< Path to the output file
    std::string message_;
    std::string shared_lib_path_;           //< Path to shared library
    std::string controller_function_name_;  //< Name of the controller function in the shared library

    util::dylib lib_;  //< Handle to the shared library

    /// Function pointer to the controller function in the shared library
    std::function<void(float*, int*, const char*, const char*, const char*)> controller_function_;
};

}  // namespace openturbine::util
