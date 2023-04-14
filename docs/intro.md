# OpenTurbine: A flexible multibody structural dynamics code for wind turbines

:::{note}
This page is under construction. Please check back here throughout FY23
for updates, and see development activity at https://github.com/exawind/openturbine.
:::

OpenTurbine is a new, open-source wind turbine structural dynamics simulation
code designed to meet the research needs of Wind Energy Technologies Office (WETO)
and the broader wind energy community for land-based and offshore wind turbines.
OpenTurbine will provide high-fidelity, highly performant structural dynamics
models that can couple with low-fidelity aerodynamic/hydrodynamic models like those
in [OpenFAST](https://github.com/OpenFAST/openfast), as well as high-fidelity
computational fluid dynamics (CFD) models like those in the WETO and Office
of Science supported [ExaWind](https://github.com/Exawind) code suite.

Following describes the high-level development objectives conceived for OpenTurbine:
- OpenTurbine will follow modern software development best practices. The
development process will require test-driven development (TDD), version control,
hierarchical automated testing, and continuous integration leading to a
robust development environment.
- The core data structures will be memory efficient and will enable vectorization
and parallelization at multiple levels.
- They will be data-oriented to exploit methods for accelerated computing including
high utilization of chip resources (e.g., single instruction multiple data i.e. SIMD),
parallelization through GPGPUs or other hardware, and support for memory-efficient
architectures.
- The computational algorithms will incorporate robust open-source libraries for
mathematical operations, resource allocation, and data management.
- The API design will consider multiple stakeholder needs and ensure
integration with existing and future ecosystems for data science, machine learning,
and AI.
- OpenTurbine will be written in modern C++ and will leverage [Kokkos](https://github.com/kokkos/kokkos)
as its performance-portability library with inspiration from the ExaWind stack.


```{tableofcontents}
```
