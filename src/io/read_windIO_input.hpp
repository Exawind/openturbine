#pragma once

#include <any>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace openturbine::io {

// Print a YAML node to the console
void print_node(const YAML::Node& node, int indent = 0);

}  // namespace openturbine::io
