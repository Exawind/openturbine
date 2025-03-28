Developer Tools
===============

This page describes tools used in the development of OpenTurbine. These should
generally be adopted by all developers so that expectations and systems are
aligned.

clang-format
------------

`ClangFormat <https://clang.llvm.org/docs/ClangFormat.html>`_ is used for
linting to enforce a consistent code style. It can be installed with most package
managers. The syntax to run the linter is given below.

.. code-block:: bash

   # Lint in place all .cpp files in the src directory
   clang-format -i src/*.cpp

   # Show changes in stdout for all header files in the utilities directory
   clang-format -i src/utilities/*.H

ClangFormat is configured by the ``.clang-format``. If the tool is run from the
top directory, it will automatically detect and load the settings in the
configuration file.

.. note::

   The CI system runs this linter and any required changes will cause the system
   to fail.
