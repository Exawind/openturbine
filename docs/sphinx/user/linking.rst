Linking To OpenTurbine
======================

OpenTurbine is fully discoverable using CMake's ``find_package`` utility.
Simply add the following line to your ``CMakeLists.txt`` file.

.. code-block:: cmake
   
    find_package(OpenTurbine REQUIRED)

This utility will search your path for your OpenTurbine installation and load its target information.
To link against OpenTurbine, add to your ``CMakeLists.txt`` the line

.. code-block:: cmake

   target_link_libraries(my_executable PRIVATE OpenTurbine::openturbine_library)

This line will link to OpenTurbine and all of its dependencies - there is no need to include any transitive dependencies, such as Kokkos, explicitly.
