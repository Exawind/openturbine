# OpenTurbine API documentation {#mainpage}

This document is intended for developers who want to understand the C++ code
structure and modify the codebase and, therefore, assumes that the reader is
familiar with the installation, compilation, and execution steps. If you are new to
OpenTurbine and haven't installed/used OpenTurbine previously, we recommend starting
with the [user manual](https://exawind.github.io/openturbine/) that provides a detailed
overview of the installation process as well as general usage.

## How to use this API guide?

TODO

### Source code organization

Upon successful download/clone, the base repository (`openturbine`) has source code
organized in subdirectories described below:

- `cmake` -- Functions and utilities used during CMake configuration phase
- `docs` -- User manual (Sphinx-based) and Doxygen files
- `src` -- C++ source files. All code is located within this directory and is
  organized into sub-directories that represent the different parts of the codebase.
- `test` -- Divided into unit-tests and regression tests, contains test files for
  the codebase

When developing new features, we strongly recommend creating a unit-test and
develop features incrementally and testing as you add capabilities. Unit-tests
are also a good way to explore the usage of individual components of the code.

## Contributing

OpenTurbine is an open-source code and we welcome contributions from the community.
Please consult the [developer
documentation](https://exawind.github.io/openturbine/developer/index.html) section
of the user manual to learn about the process of submitting code enhancements,
bug-fixes, documentation updates, etc.

## License

OpenTurbine is licensed under MIT license. Please see the
[LICENSE](https://github.com/Exawind/openturbine/blob/main/LICENSE) included in
the source code repository for more details.
