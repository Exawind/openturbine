import os
import yaml

"""
Script to convert a YAML schema to C++ structs. The schema is read from the windIO schema file and the structs are written to a C++ file.
"""

# type Definitions map[string]yaml.Node
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
        modified = modified.replace('/', 'DividedBy')

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

    # Check if the struct is already in the map
    if s.name in struct_map:
        print(f"Struct {s.name} already exists")
        # Now check if the struct in the map has the same fields as the struct we are trying to build
        same = True
        for field in s.fields:
            if field not in struct_map[s.name].fields:
                # If the field is not in the struct in the map, the struct is not the same
                same = False
                break
        if same:
            # If the struct is the same, return
            print(f"Skipping {s.name}")
            return

        # If the struct is not the same, add a suffix to the name by keeping track of the number of structs with the same name
        i = 1
        while f"{s.name}_{i}" in struct_map:
            i += 1
        s.name = f"{s.name}_{i}"

    # Add struct to map
    struct_map[s.name] = s

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


def get_ref(ref: str, definitions: Definitions, struct_map: dict[str, Struct]) -> tuple[str, Schema]:
    ref = ref.removeprefix("#/definitions/")
    parts = ref.split("/")
    name = parts[0]
    node = definitions[name]
    definition_map = {k: Schema(v) for k, v in node.items()}
    schema = definition_map[parts[1]]
    return name, schema


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

    # If the schema has multiple types, use the first one (for now)
    if schema.one_of:
        schema = schema.one_of[0]

    # Set the type based on the schema type
    if schema.type == 'object': # If the field is an object, build a Struct for it
        field.type = field.name
        build_structs(Struct(field.name, schema.description), schema, definition_map, struct_map)
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

    # print the detected type
    #print(f"{field.name}: {field.type}")


def main():
    """
    Main function to build the structs from the schema
    """
    with open("/Users/fbhuiyan/dev/openturbine/src/utilities/scripts/IEAontology_schema.yaml", 'r') as file:
        data = yaml.safe_load(file)

    root = Schema(data)

    # Build the structs from the schema
    struct_map = {}
    build_structs(Struct("Turbine"), root, root.definitions, struct_map)

    struct_names = sorted(struct_map.keys())

    # Write structs to file
    with open("/Users/fbhuiyan/dev/openturbine/src/utilities/scripts/structs_v2.cpp", 'w') as file:
        # Write includes
        file.write("#include <string>\n#include <vector>\n\n")

        # Write structs
        for struct_name in struct_names:
            s = struct_map[struct_name]
            file.write(f"// {s.name} - {s.desc}\n")
            file.write(f"struct {s.name} {{\n") # Write struct name

            # Write fields
            for f in s.fields:
                file.write(f"    {f.type} {f.name_yaml}; // {f.desc}\n") # Write field
            file.write("};\n\n")

if __name__ == "__main__":
    # test_get_ref()
    # test_build_type()
    main()

# def test_build_type():
#     """
#     Test the build_type function
#     """
#     # Test object
#     field = Field('test_object', 'test_object', '', '')
#     schema = Schema({'type': 'object', 'properties': {'a': {'type': 'string'}}})
#     definition_map = {}
#     struct_map = {}
#     build_type(field, schema, definition_map, struct_map)
#     assert field.type == 'test_object'
#     assert 'test_object' in struct_map

#     # Test string
#     field = Field('test_string', 'test_string', '', '')
#     schema = Schema({'type': 'string'})
#     definition_map = {}
#     struct_map = {}
#     build_type(field, schema, definition_map, struct_map)
#     assert field.type == 'std::string'
#     assert 'test_object' not in struct_map

#     # Test number
#     field = Field('test', 'test', '', '')
#     schema = Schema({'type': 'number'})
#     definition_map = {}
#     struct_map = {}
#     build_type(field, schema, definition_map, struct_map)
#     assert field.type == 'double'
#     assert 'test_object' not in struct_map

#     # Test integer
#     field = Field('test', 'test', '', '')
#     schema = Schema({'type': 'integer'})
#     definition_map = {}
#     struct_map = {}
#     build_type(field, schema, definition_map, struct_map)
#     assert field.type == 'int'
#     assert 'test_object' not in struct_map

#     # Test boolean
#     field = Field('test', 'test', '', '')
#     schema = Schema({'type': 'boolean'})
#     definition_map = {}
#     struct_map = {}
#     build_type(field, schema, definition_map, struct_map)
#     assert field.type == 'bool'
#     assert 'test_object' not in struct_map

#     # Test array
#     field = Field('test', 'test', '', '')
#     schema = Schema({'type': 'array', 'items': {'type': 'string'}})
#     definition_map = {}
#     struct_map = {}
#     build_type(field, schema, definition_map, struct_map)
#     assert field.type == 'std::vector<std::string>'
