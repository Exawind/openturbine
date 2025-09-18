# Rigid Body with Three Springs

This simple test shows how to run a simulation of a rigid body affected by an artificial buoyancy force and tied down by three linear springs.
It uses Kynema's high level "CFD" interface for easy definition of this common problem configuration.

To build this test, first ensure that Kynema has been installed somewhere that is discoverable by CMake (if using the Spack pacakge manager, run `spack load kynema`).
Next, create a build directory and from there run `cmake ../` and `make`.
Note that it does not do any IO itself, so you will have to add that to inspect the results yourself.
