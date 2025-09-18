# IEA15MW Turbine

This test shows how to run a simulation of a full wind turbine model based on a realistic specification of the IEA15MW turbine.
It uses Kynema's high level "Turbine" interface for easy definition of this common problem configuration.

To build this test, first ensure that Kynema has been installed somewhere that is discoverable by CMake (if using the Spack pacakge manager, run `spack load kynema`).
Next, create a build directory and from there run `cmake ../` and `make`.
Note that it does not do any IO itself, so you will have to add that to inspect the results yourself.
When running this test, you will need to be in the same directory as the included YAML input file.
