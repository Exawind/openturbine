# OpenTurbine

**OpenTurbine development began in 2023, is ongoing, and we aim for beta release in 2025**

OpenTurbine is an open-source flexible-multibody-dynamics simulation
code designed to meet the research needs of Wind Energy Technologies Office (WETO)
and the broader wind energy community for land-based and offshore wind turbines.
OpenTurbine provides high-fidelity, highly performant structural dynamics
models that can couple with low-fidelity aerodynamic/hydrodynamic models like those
in [OpenFAST](https://github.com/OpenFAST/openfast), as well as high-fidelity
computational fluid dynamics (CFD) models like those in the WETO and Office
of Science supported [ExaWind](https://github.com/Exawind) code suite.

The following describes the high-level development objectives conceived for OpenTurbine:
- OpenTurbine will follow modern software development best practices, 
including test-driven development (TDD), version control,
hierarchical automated testing, and continuous integration (CI) for a
robust development environment.
- The core data structures are memory efficient and enable vectorization
and parallelization at multiple levels.
- Data structures are data-oriented to exploit methods for accelerated computing including
high utilization of chip resources (e.g., single instruction multiple data i.e. SIMD),
parallelization through GP-GPUs or other hardware, and support for memory-efficient
architectures.
- The computational algorithms incorporate robust open-source libraries for
mathematical operations, resource allocation, and data management.
- The API design considers multiple stakeholder needs and ensure
integration with existing and future ecosystems for data science, machine learning,
and AI.
- OpenTurbine is written in modern C++ and leverages [Kokkos](https://github.com/kokkos/kokkos)
as its performance-portability library with inspiration from the ExaWind stack.

## Development support

OpenTurbine is primarily developed with the support of the U.S. Department of Energy (DOE) and is part of the [WETO Software Stack](https://nrel.github.io/WETOStack). For more information and other integrated modeling software, see:
- [Portfolio Overview](https://nrel.github.io/WETOStack/portfolio_analysis/overview.html)
- [Entry Guide](https://nrel.github.io/WETOStack/_static/entry_guide/index.html)
- [High-Fidelity Modeling Workshop](https://nrel.github.io/WETOStack/workshops/user_workshops_2024.html#high-fidelity-modeling)

Support was also provided by the DOE Office of Science FLOWMAS Energy Earthshot Research Center.

[Software Release](https://www.osti.gov/biblio/1908664)

[Documentation](https://exawind.github.io/openturbine/)

Send questions to michael.a.sprague@nrel.gov, OpenTurbine Principal Investigator.
