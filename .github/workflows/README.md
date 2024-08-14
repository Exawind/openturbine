# GitHub Workflow Definitions for CI

This collection of GitHub workflow definitions is designed for Continuous Integration (CI) and is organized as follows:

- **correctness-linux.yaml**: Builds and runs the entire test suite on `ubuntu-latest` using both _gcc_ and _clang_ compilers. Both `Debug` and `Release` builds are tested, with `AddressSanitizer`, `LeakSanitizer`, and `UndefinedBehaviorSanitizer` enabled.  The CppCheck and ClangTidy static analysis tools are both enabled for all configurations.

- **correctness-macos.yaml**: Builds and runs the entire test suite on `macos-latest` using the _clang_ compiler, both the default version provided by Apple and the latest version. Similar to the Linux workflow, both `Debug` and `Release` builds are tested, with `AddressSanitizer` and `UndefinedBehaviorSanitizer` enabled.  The CppCheck and ClangTidy static analysis tools are both enabled for all configurations.

- **formatting.yaml**: Checks formatting for all source, header, and test files. Note that this workflow does not automatically fix formatting errors; developers are responsible for correcting any issues detected.
