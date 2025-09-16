User Manual
=============

This manual will describe how to build and link against OpenTurbine.
It also provides several examples for how to use OpenTurbine to model and solve structural dynamics problems.
To find working versions of each of these problems, see the `tests/documentation_tests/` folder.
There, you will also find simple CMake scripts demonstrating how to link against OpenTurbine, as described in this manual.
These examples are meant to be linked against an OpenTurbine installation and can serve as the basis for specifiying your own problem using our interfaces.
For more detailed build instructions, see the README files in their respective folders.

.. toctree::
   :glob:
   :maxdepth: 2

   compiling.rst
   linking.rst
   vtk_visualization.rst
   floating_platform.rst
   turbine.rst
   heavy_top.rst
   spring_mass.rst
   three_blade.rst
