#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "utilities/controllers/discon.hpp"
#include "utilities/controllers/turbine_controller.hpp"
#include "vendor/dylib/dylib.hpp"

#include "OpenTurbine_config.h"

namespace openturbine::tests {

TEST(ROSCO_Controller, initialize) {
    const auto shared_lib_path = std::string{static_cast<const char*>(OpenTurbine_ROSCO_LIBRARY)};
    const auto controller_function_name = std::string{"DISCON"};

    auto controller = util::TurbineController(
        shared_lib_path, controller_function_name, "./IEA-15-240-RWT/DISCON.IN", ""
    );

    controller.io.status = 0;
    controller.io.time = 0.;
    controller.io.dt = 0.01;
    controller.io.rotor_speed_actual = 5.;

    controller.CallController();
}

}  // namespace openturbine::tests
