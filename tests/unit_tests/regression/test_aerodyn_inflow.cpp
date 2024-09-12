#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/vendor/dylib/dylib.hpp"

namespace openturbine::tests {

#ifdef OpenTurbine_BUILD_OPENFAST_ADI
TEST(AerodynInflowTest, ADI_C_PreInit) {
    // Use dylib to load the dynamic library and get access to the aerodyn inflow c binding functions
    auto path = std::string{"libaerodyn_inflow_c_binding."};
#ifdef __APPLE__
    path += "dylib";
#elif __linux__
    path += "so";
#else  // Windows
    path += "dll";
#endif
    const util::dylib lib(path, util::dylib::no_filename_decorations);
    auto ADI_C_PreInit = lib.get_function<void(int*, int*, int*, int*, char*)>("ADI_C_PreInit");

    // Call ADI_C_PreInit routine and expect the following outputs
    int numTurbines{1};               // input: Number of turbines
    int transposeDCM{1};              // input: Transpose the direction cosine matrix
    int debuglevel{0};                // input: Debug level
    int error_status_c{0};            // output: error status
    char error_message_c[] = {'\0'};  // output: error message
    ADI_C_PreInit(
        &numTurbines, &transposeDCM, &debuglevel, &error_status_c,
        static_cast<char*>(error_message_c)
    );

    EXPECT_EQ(numTurbines, 1);
    EXPECT_EQ(transposeDCM, 1);
    EXPECT_EQ(debuglevel, 0);
    EXPECT_EQ(error_status_c, 0);
    EXPECT_STREQ(static_cast<char*>(error_message_c), "");
}
#endif

}  // namespace openturbine::tests