#include "src/io/read_windIO_input.hpp"

#include <yaml-cpp/yaml.h>

namespace openturbine::io {

void print_node(const YAML::Node& node, int indent) {
    for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
        for (int i = 0; i < indent; ++i) {
            std::cout << "  ";
        }
        if (it->second.IsScalar()) {
            std::cout << it->first.as<std::string>() << ": " << it->second.as<std::string>()
                      << std::endl;
        } else if (it->second.IsSequence()) {
            std::cout << it->first.as<std::string>() << ": [";
            for (size_t i = 0; i < it->second.size(); ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << it->second[i].as<std::string>();
            }
            std::cout << "]" << std::endl;
        } else if (it->second.IsMap()) {
            std::cout << it->first.as<std::string>() << ":" << std::endl;
            print_node(it->second, indent + 1);
        }
    }
}

}  // namespace openturbine::io
