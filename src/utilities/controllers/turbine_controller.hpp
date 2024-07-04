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
        std::string shared_lib_path, std::string controller_name,
        std::string controller_function_name, std::string accINFILE = "", std::string avcOUTNAME = ""
    )
        : shared_lib_path_(shared_lib_path),
          controller_name_(controller_name),
          controller_function_name_(controller_function_name) {
        // Copy the input file and output name to the class attributes
        std::copy(accINFILE.begin(), accINFILE.end(), this->accINFILE_);
        std::copy(avcOUTNAME.begin(), avcOUTNAME.end(), this->avcOUTNAME_);
        this->swap_ = reinterpret_cast<SwapStruct*>(this->avrSWAP_);
    }

    // Get the pointer to the avrSWAP array
    SwapStruct* GetSwap() { return this->swap_; }

    // Method to call the controller function from the shared library
    void CallController() {
        // Logic to call the DLL and get the results
        util::dylib lib(this->shared_lib_path_, this->controller_name_);
        lib.get_function<
            void(float*, int&, const char*, const char*, const char*)>(this->controller_function_name_)(
            this->avrSWAP_, this->aviFAIL_, this->accINFILE_, this->avcOUTNAME_, this->avcMSG_
        );

        // Handle the errors coming out of the shared library
        if (this->aviFAIL_ < 0) {
            throw std::runtime_error(this->avcMSG_);
        } else if (this->aviFAIL_ > 0) {
            std::cout << "Warning: " << this->avcMSG_ << std::endl;
        }
    }

private:
    // Declare the attributes required to call the shared library (typically implemented in C
    // utilizing the Bladed API)
    float avrSWAP_[81] = {0.};
    int aviFAIL_;
    char accINFILE_[256];
    char avcOUTNAME_[256];
    char avcMSG_[1024];

    // Store the shared library information
    std::string shared_lib_path_;
    std::string controller_name_;
    std::string controller_function_name_;

    // Declare a raw pointer to the avrSWAP array
    // TODO Make it a smart pointer
    SwapStruct* swap_;
};

}  // namespace openturbine::util
