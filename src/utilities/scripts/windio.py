import argparse
import yaml

"""
Script to convert a YAML schema to C++ structs. The schema is read from the windIO schema file and the structs are written to a C++ file.
"""

Definitions = dict[str, yaml.Node]

class Schema:
    def __init__(self, data):
        self.description = data.get('description', '')
        self.type = data.get('type', '')
        self.properties = {k: Schema(v) for k, v in data.get('properties', {}).items()}
        self.items = Schema(data['items']) if 'items' in data else None
        self.reference = data.get('$ref', '')
        self.one_of = [Schema(i) for i in data.get('oneOf', [])]
        self.default = data.get('default', None)
        self.unit = data.get('unit', '')
        self.required = data.get('required', [])
        self.optional = data.get('optional', [])
        self.specification = data.get('$schema', '')
        self.id = data.get('$id', '')
        self.title = data.get('title', '')
        # Get definitions as yaml.Node
        self.definitions = {k: v for k, v in data.get('definitions', {}).items()}
        self.generated = False


class Field:
    def __init__(self, name, name_yaml, type, desc):
        self.name = name
        self.name_yaml = name_yaml
        self.type = type
        self.desc = desc

    def __eq__(self, other):
        return self.name == other.name and self.type == other.type and self.desc == other.desc


class Struct:
    def __init__(self, name, desc=''):
        self.name = name
        self.name_yaml = name
        self.desc = desc
        self.fields = []


def modify_name(snake_str: str) -> str:
    """
    Converts a snake_case string to PascalCase string.
    Removes spaces and replaces forward slashes with 'DividedBy'

    Args:
        snake_str (str): The snake_case string to convert

    Returns:
        str: The camelCase string
    """
    # snake_case to PascalCase
    components = snake_str.split('_')
    modified = ''.join(x.title() for x in components)

    # remove spaces
    modified = modified.replace(' ', '')

    # replace forward slashes with 'DividedBy'
    if '/' in modified:
        modified = modified.replace('/', 'DividedBy')

    return modified

def modify_variable_name(snake_str: str) -> str:
    # remove spaces
    modified = snake_str.replace(' ', '')

    # replace forward slashes with 'DividedBy'
    if '/' in modified:
        modified = modified.replace('/', '_divided_by_')

    return modified


def build_structs(s: Struct, struct_schema: Schema, definition_map: dict, struct_map: dict[str, Struct]) -> None:
    """
    Builds the struct based on the schema

    Args:
        s (Struct): The struct to build
        struct_schema (Schema): The schema of the struct
        definition_map (dict): The definitions in the schema
        struct_map (dict[str, Struct]): The structs that have been built

    Returns:
        None
    """
    # If struct has been generated, return
    if struct_schema.generated:
        return
    # Mark struct as generated
    struct_schema.generated = True

    # Check if the struct is already in the map and create test struct to compare
    if s.name in struct_map:
        # create a test field
        s_test = Struct(s.name, s.desc)
        for field_name, field_schema in struct_schema.properties.items():
            test_field = Field(
                name=modify_name(field_name),
                name_yaml=modify_variable_name(field_name),
                type=field_schema.type,
                desc=field_schema.description,
            )
            build_type(test_field, field_schema, definition_map, struct_map)
            s_test.fields.append(test_field)

        # Now check if the struct in the map has the same fields as the struct we are trying to build
        same = True
        for field in s_test.fields:
            if field not in struct_map[s.name].fields:
                # If the field is not in the struct in the map, the struct is not the same
                same = False
                break
        if same:
            return

        # If the struct is not the same, add a suffix to the name by keeping track of the number of structs with the same name
        i = 1
        while f"{s.name}_{i}" in struct_map:
            i += 1
        s.name = f"{s.name}_{i}"

    # Loop through properties in object schema and create fields
    for field_name, field_schema in struct_schema.properties.items():
        field = Field(
            name=modify_name(field_name),
            name_yaml=modify_variable_name(field_name),
            type=field_schema.type,
            desc=field_schema.description,
        )

        # Build the type of the field
        build_type(field, field_schema, definition_map, struct_map)

        # Add the field to the struct
        s.fields.append(field)

    # Add struct to map
    struct_map[s.name] = s


def get_ref(ref: str, definitions: Definitions, struct_map: dict[str, Struct]) -> tuple[str, Schema]:
    ref = ref.removeprefix("#/definitions/")
    parts = ref.split("/")
    name = parts[0]
    node = definitions[name]
    definition_map = {k: Schema(v) for k, v in node.items()}
    schema = definition_map[parts[1]]
    return name, schema


def set_type(field: Field, schema: Schema, definition_map: dict, struct_map: dict[str, Struct]) -> None:
    """
    Sets the type of a field based on the schema type

    Args:
        field (Field): The field to set the type of
        schema (Schema): The schema of the field
        definition_map (dict): The definitions present in the schema
        struct_map (dict[str, Struct]): The structs that have already been built based on the schema

    Returns:
        None
    """
    if schema.type == 'object':
        field.type = field.name
        s = Struct(field.name, schema.description)
        build_structs(s, schema, definition_map, struct_map)
        if s.name != field.type:
            field.type = s.name
    elif schema.type == 'string':
        field.type = 'std::string'
    elif schema.type == 'number':
        field.type = 'double'
    elif schema.type == 'integer':
        field.type = 'int'
    elif schema.type == 'boolean':
        field.type = 'bool'
    elif schema.type == 'array':
        if not schema.items:
            raise ValueError(f"{field.name}: array without item spec")
        field.type = schema.items.type
        build_type(field, schema.items, definition_map, struct_map)
        field.type = f"std::vector<{field.type}>"
    else:
        raise ValueError(f"Unknown type '{schema.type} - {schema}'")


def build_type(field: Field, schema: Schema, definition_map: dict, struct_map: dict[str, Struct]) -> None:
    """
    Determines the type of a field based on the schema and builds the Struct/Class if necessary.
    - If the field is an object, a new Struct is created for it
    - If the field is a string, the type is set to std::string
    - If the field is a number, the type is set to double
    - If the field is an integer, the type is set to int
    - If the field is a boolean, the type is set to bool
    - If the field is an array, the type is set to the item type followed by []

    Args:
        field (Field): The field to build
        schema (Schema): The schema of the field
        definition_map (dict): The definitions present in the schema
        struct_map (dict[str, Struct]): The structs that have already been built based on the schema

    Returns:
        None
    """
    # If this field is a reference to a definition we need to get the schema for that definition
    if schema.reference:
        if schema.type: # If the field specifies both $ref and type, raise an error
            raise ValueError(f"{field.name} specifies $ref and type")
        _, schema = get_ref(schema.reference, definition_map, struct_map)

    # Set the type to object if no type is specified but properties are present
    if not schema.type and schema.properties:
        schema.type = 'object'

    # If the schema has multiple types, use std::variant to represent them
    if schema.one_of:
        field.type = "std::variant<"
        for i, s in enumerate(schema.one_of):
            field_dummy = Field("", "", "", "")
            set_type(field_dummy, s, definition_map, struct_map)
            field.type += field_dummy.type
            if i < len(schema.one_of) - 1:
                field.type += ", "
        field.type += ">"
        return

    # Set the type based on the schema type using the set_type function
    set_type(field, schema, definition_map, struct_map)


def set_parse_function(field: Field) -> str:
    """
    Sets the parse function for the field based on the type of the field

    Args:
        field (Field): The field to set the parse function for

    Returns:
        str: The parse function as a string
    """
    if field.type == "double":
        return f"    {field.name_yaml} = node[\"{field.name_yaml}\"] ? node[\"{field.name_yaml}\"].as<double>() : 0.;\n"
    elif field.type == "int":
        return f"    {field.name_yaml} = node[\"{field.name_yaml}\"] ? node[\"{field.name_yaml}\"].as<int>() : 0;\n"
    elif field.type == "bool":
        return f"    {field.name_yaml} = node[\"{field.name_yaml}\"] ? node[\"{field.name_yaml}\"].as<bool>() : false;\n"
    elif field.type == "std::string":
        return f"    {field.name_yaml} = node[\"{field.name_yaml}\"] ? node[\"{field.name_yaml}\"].as<std::string>() : \"\";\n"
    elif field.type.startswith("std::vector"):
        # We need to handle the case where the field is an array of objects
        if field.type.removeprefix("std::vector<").removesuffix(">") not in ["double", "int", "bool", "std::string"]:
            # if the field type starts with std::vector and is not an object, we can parse it directly
            if "std::vector" in field.type.removeprefix("std::vector<").removesuffix(">"):
                parse_string = ""
                parse_string += f"    if (node[\"{field.name_yaml}\"]) {{\n"
                parse_string += f"        for (const auto& item : node[\"{field.name_yaml}\"]) {{\n"
                parse_string += f"            {field.name_yaml}.push_back(item.as<{field.type.removeprefix('std::vector<').removesuffix('>')}>());\n"
                parse_string += f"        }}\n"
                parse_string += f"    }}\n"
                return parse_string
            # Assume the field is an array of objects and we need to parse each object
            parse_string = ""
            parse_string += f"    if (node[\"{field.name_yaml}\"]) {{\n"
            parse_string += f"        for (const auto& item : node[\"{field.name_yaml}\"]) {{\n"
            parse_string += f"            {field.type.removeprefix('std::vector<').removesuffix('>')} x;\n"
            parse_string += f"            x.parse(item);\n"
            parse_string += f"            {field.name_yaml}.push_back(x);\n"
            parse_string += f"        }}\n"
            parse_string += f"    }}\n"
            return parse_string
        else:
            return f"    {field.name_yaml} = node[\"{field.name_yaml}\"] ? node[\"{field.name_yaml}\"].as<{field.type}>() : {field.type}();\n"
    # Assume everything else is an object
    else:
        return f"    if (node[\"{field.name_yaml}\"]) {{\n        {field.name_yaml}.parse(node[\"{field.name_yaml}\"]);\n    }}\n"


def build_parse_function(s: Struct) -> str:
    """
    Builds the parse function for the struct

    Args:
        s (Struct): The struct to build the parse function for

    Returns:
        str: The parse function as a string
    """
    parse_function = f"void parse(const YAML::Node& node) {{\n"
    for field in s.fields:
        if field.type.startswith("std::variant"):
            # TODO
            # - We are assuming all std::variants depend on the "orth" field. This may not be the case.
            # - Following is hardcoded for two types in the variant. Need to generalize for more types.

            # if field is either of [E, G, nu, alpha, Xt, Xc, Xy, S] then it depends on "orth" field
            if field.name_yaml in ["E", "G", "nu", "alpha", "Xt", "Xc", "Xy", "S"]:
                parse_function += f"    if (!orth) {{\n"
                for i, s in enumerate(field.type.removeprefix("std::variant<").removesuffix(">").split(", ")):
                    field_dummy = Field("", "", "", "")
                    field_dummy.type = s
                    field_dummy.name_yaml = f"{field.name_yaml}"
                    parse_function += set_parse_function(field_dummy)
                    if i < len(field.type.removeprefix("std::variant<").removesuffix(">").split(", ")) - 1:
                        parse_function += f"    }}\n"
                        parse_function += f"    else {{\n"
                parse_function += "    }\n"
            # Field does not depend on "orth" field - parse as a double
            else:
                parse_function += f"    {field.name_yaml} = node[\"{field.name_yaml}\"] ? node[\"{field.name_yaml}\"].as<double>() : 0.;\n"
        else:
            parse_function += set_parse_function(field)

    parse_function += "}\n"
    return parse_function


def main():
    """
    Main function to build the structs from the schema
    """
    parser = argparse.ArgumentParser(description="Convert a YAML schema to C++ structs")
    parser.add_argument("input_file", type=argparse.FileType('r'), help="The YAML schema file e.g. IEAontology_schema.yaml")
    parser.add_argument("output_file", type=argparse.FileType('w'), help="The output C++ file e.g. IEAontology_structs.cpp")
    args = parser.parse_args()

    with open(args.input_file.name, 'r') as file:
        data = yaml.safe_load(file)

    root = Schema(data)

    # Build the structs from the schema
    struct_map = {}
    build_structs(Struct("Turbine"), root, root.definitions, struct_map)

    struct_names = struct_map.keys()

    # Write structs to file
    with open(args.output_file.name, 'w') as file:
        # Write includes
        file.write("#include <string>\n#include <vector>\n\n")

        # Write structs
        for struct_name in struct_names:
            s = struct_map[struct_name]
            file.write(f"// {s.desc}\n" if s.desc else f"// {s.name}\n")
            file.write(f"struct {s.name} {{\n") # Write struct name

            # Write fields
            for f in s.fields:
                file.write(f"    {f.type} {f.name_yaml};{f' // {f.desc}' if f.desc else ''}\n")

            # Write the parse function
            file.write(f"\n    {build_parse_function(s)}\n")

            file.write("};\n\n")

if __name__ == "__main__":
    main()
