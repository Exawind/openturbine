#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/io/read_windIO_input.hpp"
#include "src/io/windio_mapped_structs.cpp"

namespace openturbine::restruct_poc::tests {

TEST(ParserTest, ReadYAMLNode) {
    try {
        // Load the YAML file
        YAML::Node config =
            YAML::LoadFile("/Users/fbhuiyan/dev/openturbine/src/utilities/scripts/config.yaml");

        // Print the contents of the YAML file
        io::print_node(config);
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading YAML file: " << e.what() << std::endl;
    }
}

// Write a test for parsing Turbine from IEA 15 MW yaml file
// TEST(ParserTest, ParseIEA15MW) {
//     try {
//         // Load the YAML file
//         YAML::Node config =
//             YAML::LoadFile("/Users/fbhuiyan/dev/openturbine/src/io/IEA-15-240-RWT.yaml");

//         // Parse the top-level node
//         Turbine turbine;
//         turbine.parse(config);

//         // Add some assertions
//         ASSERT_EQ(turbine.name, "IEA 15MW Offshore Reference Turbine, with taped chord tip
//         design");

//         ASSERT_EQ(turbine.assembly.turbine_class, "I");
//         ASSERT_EQ(turbine.assembly.turbulence_class, "B");
//         ASSERT_EQ(turbine.assembly.drivetrain, "direct_drive");
//         ASSERT_EQ(turbine.assembly.rotor_orientation, "Upwind");
//         ASSERT_EQ(turbine.assembly.number_of_blades, 3);
//         ASSERT_EQ(turbine.assembly.hub_height, 150.);
//         ASSERT_EQ(turbine.assembly.rotor_diameter, 241.94);
//         ASSERT_EQ(turbine.assembly.rated_power, 15.e+6);
//         ASSERT_EQ(turbine.assembly.lifetime, 25.0);

//         // Assert there are 11 materials
//         ASSERT_EQ(turbine.materials.size(), 11);
//     } catch (const YAML::Exception& e) {
//         std::cerr << "Error loading YAML file: " << e.what() << std::endl;
//     }
// }

}  // namespace openturbine::restruct_poc::tests
