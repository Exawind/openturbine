#include <gtest/gtest.h>

#include "test_utilities.hpp"

#include "src/io/read_windIO_input.hpp"

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

}  // namespace openturbine::restruct_poc::tests
