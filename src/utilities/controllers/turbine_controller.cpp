#include "src/utilities/controllers/turbine_controller.hpp"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace openturbine::util {

TurbineController::TurbineController(
    std::string shared_lib_path, std::string controller_function_name, std::string input_file_path,
    std::string output_file_path
)
    : input_file_path_(input_file_path),
      output_file_path_(output_file_path),
      shared_lib_path_(shared_lib_path),
      controller_function_name_(controller_function_name),
      lib_(shared_lib_path, util::dylib::no_filename_decorations) {
    // Make sure we have a valid shared library path + controller function name
    try {
        lib_.get_function<void(
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

void TurbineController::CallController() {
    this->lib_.get_function<
        void(float*, int*, const char*, const char*, const char*)>(this->controller_function_name_)(
        this->swap_array_, &this->status_, this->input_file_path_.c_str(),
        this->output_file_path_.c_str(), this->message_.c_str()
    );

    if (this->status_ < 0) {
        throw std::runtime_error("Error raised in controller: " + this->message_);
    } else if (this->status_ > 0) {
        std::cout << "Warning from controller: " << this->message_ << std::endl;
    }
}

}  // namespace openturbine::util
