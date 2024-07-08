#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

#include "src/utilities/controllers/discon.h"
#include "src/vendor/dylib/dylib.hpp"

namespace openturbine::util {

/// A turbine controller class that works as a wrapper around the shared library
/// that contains the controller logic
class TurbineController {
public:
    TurbineController(
        std::string shared_lib_path, std::string controller_function_name,
        std::string input_file_path = "", std::string output_file_path = ""
    )
        : lib(shared_lib_path,
              util::dylib::no_filename_decorations),  // TODO: add error handling for library load
          input_file_path_(input_file_path),
          output_file_path_(output_file_path),
          shared_lib_path_(shared_lib_path),
          controller_function_name_(controller_function_name) {
        // Map swap array to IO structure
        this->io = reinterpret_cast<ControllerIO*>(this->swap_array);

        // Populate size of character arrays
        this->io->message_array_size = this->message_.size();
        this->io->infile_array_size = this->input_file_path_.size();
        this->io->outname_array_size = this->output_file_path_.size();

        // Get pointer to the function inside shared library
        // TODO: Add error handling because this can fail
        this->func_ptr_ = lib.get_function<void(
            float* avrSWAP, int* aviFAIL, char* const accINFILE, char* const avcOUTNAME,
            char* const avcMSG
        )>(this->controller_function_name_);
    }

    // Method to call the controller function from the shared library
    void CallController() {
        // Logic call the controller function
        this->func_ptr_(
            this->swap_array, &this->status_, this->input_file_path_.data(),
            this->output_file_path_.data(), this->message_.data()
        );

        // Handle the errors coming out of the shared library
        if (this->status_ < 0) {
            throw std::runtime_error(this->message_);
        } else if (this->status_ > 0) {
            std::cout << "Warning: " << this->message_ << std::endl;
        }
    }

    // Pointer to structure mapping swap array to named fields
    ControllerIO* io;

private:
    // Shared library handle
    util::dylib lib;

    // Pointer to controller function in dll
    void (*func_ptr_)(
        float* avrSWAP, int* aviFAIL, char* const accINFILE, char* const avcOUTNAME,
        char* const avcMSG
    ) = nullptr;

    // Declare the attributes required to call the shared library (typically implemented in C
    // utilizing the Bladed API)
    float swap_array[81] = {0.};
    int status_ = 0;
    std::string input_file_path_;
    std::string output_file_path_;
    std::string message_ = std::string(1024, ' ');

    // Store the shared library information
    std::string shared_lib_path_;
    std::string controller_function_name_;
};

}  // namespace openturbine::util
