# GitHub Workflow Definitions for CI

This collection of GitHub workflow definitions is designed for Continuous Integration (CI) and is organized as follows:

- **correctness-linux.yaml**: Builds and runs the entire test suite on `ubuntu-latest` using both _gcc_ and _clang_ compilers. Both `Debug` and `Release` builds are tested, with `AddressSanitizer` and `UndefinedBehaviorSanitizer` enabled.

- **correctness-macos.yaml**: Builds and runs the entire test suite on `macos-latest` using the _AppleClang_ compiler. Similar to the Linux workflow, both `Debug` and `Release` builds are tested, with `AddressSanitizer` and `UndefinedBehaviorSanitizer` enabled.

- **formatting.yaml**: Checks formatting for all source, header, and test files. Note that this workflow does not automatically fix formatting errors; developers are responsible for correcting any issues detected.

- **clang-tidy.yaml**: Runs the `clang-tidy` static analysis tool on all source, header, and test files.

- **cppcheck.yaml**: Runs the `CppCheck` static analysis tool on all source, header, and test files.
