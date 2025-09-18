# Three Blade Rotor

This test shows how to assemble and run a simulation of a three bladed rotor with motion prescribed to a common hub.
It uses Kynema's low level interface, which allows unlimited flexibility in problem specification.

To build this test, first ensure that Kynema has been installed somewhere that is discoverable by CMake (if using the Spack pacakge manager, run `spack load kynema`).
Next, create a build directory and from there run `cmake ../` and `make`.
Note that it does not do any IO itself, so you will have to add that to inspect the results yourself.
