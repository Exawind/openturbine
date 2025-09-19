# Kynema
[Documentation](https://kynema.github.io/kynema) | [Nightly test dashboard](http://my.cdash.org/index.php?project=Exawind) 

**Kynema development began in 2023, is ongoing, and we aim for beta release in 2025**

Kynema is an open-source flexible-multibody-dynamics simulation
code designed to meet the research needs of Wind Energy Technologies Office (WETO)
and the broader wind energy community for land-based and offshore wind turbines.
Kynema provides high-fidelity, highly performant structural dynamics
models that can couple with low-fidelity aerodynamic/hydrodynamic models like those
in [OpenFAST](https://github.com/OpenFAST/openfast), as well as high-fidelity
computational fluid dynamics (CFD) models like those in the WETO and Office
of Science supported [ExaWind](https://github.com/Exawind) code suite.

The following describes the high-level development objectives conceived for Kynema:
- Kynema will follow modern software development best practices, 
including test-driven development (TDD), version control,
hierarchical automated testing, and continuous integration (CI) for a
robust development environment.
- The core data structures are memory efficient and enable vectorization
and parallelization at multiple levels.
- Data structures are data-oriented to exploit methods for accelerated computing including
high utilization of chip resources (e.g., single instruction multiple data (SIMD) instruction sets) and
parallelization using GP-GPUs.
- The computational algorithms incorporate robust open-source libraries for
mathematical operations, resource allocation, and data management.
- The API design considers multiple stakeholder needs and ensure
integration with existing and future ecosystems for data science, machine learning,
and AI.
- Kynema is written in modern C++ and leverages [Kokkos](https://github.com/kokkos/kokkos)
as its performance-portability library with inspiration from the ExaWind stack.

## Contributing
Kynema is an open-source project and we welcome contributions from external developers.
To do so, open an issue describing your contribution and make a pull request against Kynema's main branch.
Smaller contributions are always preferred - 10 self-contained 200 line changes are easier to review and coordinate with others than one 2000 line change.

When adding a feature, make sure that it is comprehensively covered by unit tests and regression tests.
Bug fixes should be accompanied by at least one test (but possibly more) which fails without the fix but now passes.

Kynema's CI process targets a number of different configurations for MacOS and Linux, but is not fully comprehensive of the platforms we support.
In your PR, please indicate on which platforms you've tested your contribution (i.e. Linux x86 and CUDA v12).
This will let us know what other platforms we may have to test against in the review process.

## Development support

Kynema is primarily developed with the support of the U.S. Department of Energy (DOE) and is part of the [WETO Software Stack](https://nrel.github.io/WETOStack). For more information and other integrated modeling software, see:
- [Portfolio Overview](https://nrel.github.io/WETOStack/portfolio_analysis/overview.html)
- [Entry Guide](https://nrel.github.io/WETOStack/_static/entry_guide/index.html)
- [High-Fidelity Modeling Workshop](https://nrel.github.io/WETOStack/workshops/user_workshops_2024.html#high-fidelity-modeling)

Support was also provided by the DOE Office of Science FLOWMAS Energy Earthshot Research Center.

[Kynema Software Release Record SWR-23-07](https://www.osti.gov/biblio/1908664)

[Documentation](https://kynema.github.io/kynema/)

Send questions to michael.a.sprague@nrel.gov, Kynema Principal Investigator.
