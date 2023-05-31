(dev-plan)=

# OpenTurbine Development Plan

:::{warning}
This page is under construction. Please check back here throughout FY23
for updates, and see development activity at https://github.com/exawind/openturbine.
:::

## Background and overview

OpenTurbine development started in early 2023 with primary funding from the
U.S. Department of Energy (DOE) Wind Energy Technologies Office (WETO) and with
additional funding from the DOE Exascale Computing Project (ECP). It is being
developed by researchers at the National Renewable Energy Laboratory (NREL)
and the Sandia National Laboratories (SNL).

OpenTurbine is envisioned to be an open-source wind turbine structural dynamics
simulation code designed to meet the research needs of WETO and the broader
wind energy community for land-based and offshore wind turbines. OpenTurbine
will provide high-fidelity, highly performant structural dynamics models that
can couple with low-fidelity aerodynamic/hydrodynamic models like those in
[OpenFAST](https://github.com/OpenFAST/openfast), and high-fidelity computational
fluid dynamics (CFD) models like those in the WETO and Office of Science supported
[ExaWind](https://github.com/Exawind) code suite. OpenTurbine will be designed
deliberately to address shortcomings of wind turbine structural models and codes that
are critical to the success of WETO modeling efforts.

## Development priorities and use cases

*Development priorities:* Considering lessons learned from nearly a decade of
OpenFAST development, OpenTurbine will follow modern software development best
practices. The development process will require test-driven development,
version control, hierarchical automated testing, and continuous integration
leading to a robust development environment. The core data structures will be
memory efficient and will enable vectorization and parallelization at multiple
levels. They will be data-oriented in order to exploit methods for accelerated
computing including high utilization of chip resources (e.g., single
instruction multiple data [SIMD]), parallelization through GPU or other
hardware, and support for memory-efficient architectures such as the ARM-based
Apple M-series chips.

*Use-case priority:* Time-domain simulation of land-based wind turbine dynamics coupled to computational fluid dynamics for fluid-structure-interaction.

## Design drivers and considerations

OpenTurbine is a relatively small, narrowly scoped software project.
Organizationally, it is very lean with a minimal and focused development
team. These factors are critical when considering the characteristics
of the software design. The project is driven by an objective to be
sustainable, extendable, accessible, and performant for many years to come.

Generally, this software should tend toward simplicity where possible
with a consideration for how new developers, current developers
in the future, and external stakeholders will be able to understand
the nuances of the code. In short, **don't be clever** and **keep it simple**.

### Lean software

The scope of this software should be defined early in the development process,
and any expansion of the scope should be critically evaluated before accepting.
The approach to the scope of OpenTurbine should be conservative. The default
decision should be to retain the initial scope and change only if absolutely
necessary.

To limit the burden of development and responsibility of maintenance,
workflows should leverage existing software as much as possible. For instance,
input and output files should be handled by a third-party library. Similarly,
visualization of results, derivations of statistics, and math portability
are other areas where the ecosystem of open source scientific software
should be leveraged, and preference should be given to software internal to
NREL or funded by WETO.

### Modular architecture

The code architecture should be structured such that each portion of code
is responsible for a minimal unit of work. Low-level units can be combined
at an intermediate level to do more work, but the scope of each unit
throughout the architecture hierarchy should be explicitly described.
This design should ensure a modular architecture, and a test of success
is to measure the difficulty and depth of changes required to swap a
particular unit of the code.

The graphic below depicts a generic but typical data pipeline in scientific
software. A well encapsulated and modular architecture should support
something like swapping the library for handling YAML input files or
changing a solver type without modifying the modules around it.
The flow of data should be considered in discrete steps in a pipeline
rather than a monolithic system working on the data.

```{mermaid}
flowchart LR

    Start(( ))

    subgraph I/O
        direction LR
        io1{{YAML}}
        io2{{JSON}}
    end

    DataModel[[Data Model]]

    subgraph Solver
        direction LR
        solver1{{Type 1}}
        solver2{{Type 2}}
        solver3{{Type 3}}
    end

    Output[[Output Model]]

    subgraph Export
        direction LR
        export1{{Visualization}}
        export2{{ASCII}}
        export3{{Commercial}}
    end

    Finish(( ))

    Start --- I/O
    I/O --- DataModel
    DataModel --- Solver
    Solver --- Output
    Output --- Export
    Export --- Finish
```

It is critical to the sustainability and stability of OpenTurbine
to maintain independence from external software even though there
will be reliance on existing libraries for common tasks. The modular
design should include data structures and API's that are general enough
to support integration of third party libraries as well as the ability
to change any library for another that accomplishes a similar task
even if by alternative methods.

### Performance first

A key design consideration of OpenTurbine is computational efficiency
or performance. Both the quantity of work and the efficiency of data should
be considered and measured (i.e., profiled) during any development effort.
A modular architecture should support offloading computationally expensive
tasks to hardware accelerators or specialized libraries, and support
multiple options for doing so depending on user configurations and
the computational environment.

Similar to modularity in the architecture, expensive tasks should be
structured in a kernel form. This low-level design pattern combines
expensive mathematical operations into an aggregate form, and structures
them so that performance libraries or compilers can parallelize the
computation. This pattern helps to encapsulate expensive operations and
algorithms. Additionally, it follows the modular architecture design
described above in that it supports swapping accelerators or parallelization
methods.


#### Data-oriented design

OpenTurbine developers designing new algorithms and data structures
should become familiar with the concepts of [data-oriented design](https://en.wikipedia.org/wiki/Data-oriented_design),
particularly [structures of arrays vs. arrays of structures](https://stackoverflow.com/questions/17924705/structure-of-arrays-vs-array-of-structures).
The key concept of this paradigm is to structure data so that it maps
closely to the form it will be represented and used within the relevant
algorithms and processing units. Specifically, developers should choose between
a structure of arrays (SoA) and array of structures (AoS) representation.
While there may be an inclination to construct a data model for best
readability, it is important to consider the computational efficiency.
A balance must be found, and documentation for any design decision is
an important tool to resolve this tension. The graphic below illustrates the
difference between structures of arrays and arrays of structures for a data
type consisting of three components of a  location and a magnitude such as
a point in 3D space (i.e., voxel or point in a fluid domain).

```{image} images/AoS_SoA.pdf
:alt: aos_soa
:width: 400px
:align: center
```

For operations involving vector math or accessing the same attribute of many
objects, the SoA pattern typically ensures that arrays are byte-aligned
to the size of CPU registers for operations on the entire array. The effect
is that compilers will add less padding to arrays in order to ensure
alignment. In operations that access all attributes of a particular object
for a computation, the AoS pattern structures the memory in a contiguous
form.

### Accessible software

Access to the software is an important consideration for the
longevity and relevance of OpenTurbine. In short, if the software is
not accessible, it won't be used, extended, or maintained. All
development efforts should always consider user and developer accessibility
as a key driver. The distinction between "user" and "developer"
is not always clear, but accessibility efforts should address concerns
for both types of engagement with OpenTurbine.

Documentation is the primary tool for addressing accessibility. Documentation
should be considered a first-priority for general design decisions,
public API's, and input files. For internal code, new features and changes
should be well described in words, diagrams, and math in their associated
pull requests. In general, it is good practice to first describe a scope
of work outlining objectives and methods. Then, public facing code should
be prototypes in contextual workflows and internal code should be
prototypes in unit tests following typical test driven development
processes.

The high level user interface should be expressive and easily accessible
through common computational tools. For example, it is typical to include
a Python interface to compiled code for easier data generation and scripting.

## Programming language and models

OpenTurbine is envisioned with a core written in C++ and leveraging [Kokkos](https://github.com/kokkos/kokkos) as its performance-portability library with inspiration from the ExaWind
stack including [Nalu-Wind](https://github.com/Exawind/nalu-wind).

## Application Programming Interface (API)

The primary goal of the API is to provide data structures and interfaces
necessary for coupling OpenTurbine to the ExaWind CFD codes for
fluid-structure interaction simulations. For land-based wind, the interface
will be designed to couple the beam finite element models and point-mass
elements (e.g., representation of the nacelle) to a CFD mesh. We will leverage
the mesh mapping that was implemented and tested in the ExaWind codes. The
representation of the turbine geometry is handled within the fluid solver,
either as a three-dimensional surface mesh for high-fidelity geometry-resolved
simulations, or as an actuator-line for mid-fidelity simulations. Those
algorithms will work well for problems where a floating platform is represented
as a point-mass. For geometry-resolved floating-platform offshore simulations,
the API will be expanded to handle mapping from the structure surface mesh (if
solid/shell elements are used) to the CFD mesh.

## Key numerical algorithms

The models necessary for mid- to high-fidelity simulation of wind turbine
structural dynamics include linear and nonlinear finite-element models coupled
through constraints equations. For example, turbine blades may be modeled with
nonlinear beam finite elements, wherein the blade roots are constrained to
rotate with the hub. These models together constitute a set of
differential-algebraic equations (DAEs) in the time domain. We will build on the
experiences gained with OpenFAST, particularly its nonlinear beam-dynamics module,
[BeamDyn](https://github.com/OpenFAST/openfast/tree/main/modules/beamdyn).

For time integration of the index-3 DAEs, we will leverage the generalized-alpha
algorithm now established in BeamDyn which allows for user-controlled numerical
damping. See [Arnold and Brüls (2007)](https://link.springer.com/article/10.1007/s11044-007-9084-0) for details.

The primary model for the turbine blades and the tower, which are slender and flexible
structures, will be nonlinear finite elements based on the geometrically exact beam
theory (GEBT), which includes bend-twist coupling. Spatial discretization
will be based on the Legendre spectral finite elements (LSFEs; like
those in BeamDyn), but with the explicit ability to use the subset of 2-node
finite elements (a special case of LSFEs).

The choice of rotation representation is being determined. BeamDyn relies on the
Wiener-Milenkovic vectorial rotation parameterization (three rotational degrees
of freedom), which contains singularities. The team is currently investigating the
use of [quaternions](https://en.wikipedia.org/wiki/Quaternion), which are
singularity free but require four parameters to represent rotation, i.e., they do
not form a minimum set.

## Verification and validation cases

The list of verification and validation cases is a work in progress. By way of
semantics, verification cases are those for which an analytical solution exists and
formal accuracy studies can be examined.  Validation cases are those for which
we have solutions that are deemed to be better representations of reality. For
example, validation results might be from experiments or from higher-fidelity
numerical simulations such as shell or solid finite element models.

**Verification cases**
- Rigid-body dynamics: three-dimensional pendulum
- Cantilever-beam nonlinear static roll up

**Validation cases**
- Princeton beam experiment
- Twisted and curved beams where the benchmarks are highly resolved solid-element Ansys models
- IEA 15-megawatt turbine where benchmark is a highly resolved shell-element Ansys model

## Target baseline turbines

- [NREL 5-megawatt reference turbine](https://www.nrel.gov/docs/fy09osti/38060.pdf)
- [IEA 15-megawatt reference turbine](https://github.com/IEAWindTask37/IEA-15-240-RWT)

## High-level development timeline

CY = calendar year, FY = fiscal year

**CY23 Q2**: The OpenTurbine team will implement a rigid-body dynamics solver following the
concepts described above, i.e., DAE-3 coupling, quaternion-based rotation representation, and a
generalized-alpha time integrator. This proof-of-concept implementation will be made available
in the `main` branch of OpenTurbine repository and will inform the next steps in OpenTurbine
development.

**CY23 Q3**: Implement a general GEBT-based beam element that is appropriate for constrained multi-body
simulations of a wind turbine. Enable variable order finite elements and user-defined material property
definition (appropriate for modern turbine blades). Demonstrate performance for a dynamic cantilever beam
problem and compare against [BeamDyn](https://github.com/OpenFAST/openfast/tree/main/modules/beamdyn).

**CY24 Q1**: Demonstrate a wind turbine rotor simulation under prescribed loading and include code verification results and automated testing results. Include control system (e.g., [ROSCO](https://github.com/NREL/ROSCO/tree/main/ROSCO)) and pitch control of blades. Compare simulation time against an equivalent model simulated with [OpenFAST](https://github.com/OpenFAST/openfast).

**CY24 Q3**: Demonstrate a rotor simulation with fluid-structure interaction (FSI) and a pitch control system. Fluid will be represented in two ways. First, through a simple Blade Element Momentum Theory (BEMT) solver and second, where the blades are represented as actuator lines in the fluid domain (solved with the ExaWind CFD code).

**CY25 Q1**: Release a robust, well-documented, well-tested version of OpenTurbine for land-based wind turbine simulations. Demonstrate whole turbine simulation (tower, nacelle, drivetrain) capabilities with FSI coupling to ExaWind.

