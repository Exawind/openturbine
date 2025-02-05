===========
OpenTurbine
===========

`OpenTurbine <https://github.com/exawind/openturbine>`_ is an open-source wind
turbine structural dynamics simulation code designed to meet the research needs of
the Wind Energy Technologies Office (WETO) and the broader wind energy community
for both land-based and offshore wind turbines. OpenTurbine offers high-fidelity,
high-performance structural dynamics models that can integrate with low-fidelity
aerodynamic/hydrodynamic models, such as those in
`OpenFAST <https://github.com/OpenFAST/openfast>`_, as well as high-fidelity
computational fluid dynamics (CFD) models, like those in the WETO and Office of
Science-supported `ExaWind <https://github.com/Exawind>`_ code suite.

Following are the high-level development objectives of OpenTurbine:

- OpenTurbine adheres to modern software development best practices. The
  development process emphasizes test-driven development (TDD), version control,
  hierarchical automated testing, and continuous integration, leading to a robust
  development environment.
- OpenTurbine is being developed in modern C++ and leverages
  `Kokkos <https://github.com/kokkos/kokkos>`_ as its performance-portability
  library, drawing inspiration from the ExaWind stack.
- The core data structures are crafted to be memory efficient, enabling
  vectorization and parallelization at multiple levels.
- These structures are data-oriented to leverage accelerated computing methods,
  including high utilization of chip resources (e.g., single instruction multiple
  data, SIMD), parallelization through GPGPUs or other hardware, and support for
  memory-efficient architectures.
- The computational algorithms incorporate robust open-source libraries for
  mathematical operations, resource allocation, and data management.
- The API design considers the needs of multiple stakeholders, ensuring
  integration with existing and future ecosystems for data science, machine
  learning, and AI.

.. toctree::
   :maxdepth: 2

   walkthrough/index
   user/user
   theory/theory
   developer/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
