# The Heavy Top Problem

This simple test shows how to assemble and run a simulation of a rotating top modeled as a single point mass.
It uses OpenTurbine's low level interface, which allows unlimited flexibility in problem specification.


To build this test, first ensure that OpenTurbine has been installed somewhere that is discoverable by CMake (if using the Spack pacakge manager, run `spack load openturbine`).
Next, create a build directory and from there run `cmake ../` and `make`.
Note that it does not do any IO itself, so you will have to add that to inspect the results yourself.
