(dev-plan)=

# OpenTurbine Development Plan

:::{warning}
This page is under construction. Check back here throughout FY23
for updates, and see activity at https://github.com/exawind/openturbine.
:::

## Background and overview

OpenTurbine development started in early 2023 with primary funding from the
U.S. Department of Energy (DOE) Wind Energy Technologies Office and with
additional funding from the DOE Exascale Computing Project (ECP). It is being
developed by researchers at the National Rewable Energy Laboratory and Sandia
National Laboratories.

OpenTurbine is meant to be an open-source wind turbine structural dynamics
simulation code designed to meet the research needs of WETO and the broader
wind energy community for land-based and offshore wind turbines. OpenTurbine
will provide high-fidelity, highly performant structural dynamics models that
can couple with low-fidelity aerodynamic/hydrodynamic models like those in
OpenFAST, and high-fidelity computational fluid dynamics (CFD) models like
those in the WETO and Office of Science supported ExaWind code suite.
OpenTurbine will be designed to address shortcomings of wind turbine structural
models and codes that are so important to the success of WETO modeling efforts.

## Development priorities and use cases

*Development priorities:* Considering lessons learned from nearly a decade of
OpenFAST development, OpenTurbine will follow modern software development best
practices. The development process will require test-driven development,
version control, hierarchical automated testing, and continuous integration
leading to a robust development environment. The core data structures will be
memory efficient and will enable vectorization and parallelization at multiple
levels. They will be data-oriented in order exploit methods for accelerated
computing including high utilization of chip resources (e.g., single
instruction multiple data [SIMD]), parallelization through GPU or other
hardware, and support for memory-efficient architectures such as the ARM-based
Apple M-series chips. 

*Use-case priority:* Time-domain simulation of land-based wind turbine dynamics coupled to computational fluid dynamics for fluid-structure-interaction.

## Programming language and models

OpenTurbine is envisioned with a core written in C++ and leveraging Kokkos as its performance-portability library with inspiration from the ExaWind stack including Nalu-Wind CFD.

## High-level design




## Application Programming Interface (API)

The primary goal of the API is to provide data structures and interfaces
necessary for coupling OpenTurbine to the ExaWind CFD codes for
fluid-structure-interaction simulations. For land-based wind, the interface
will be designed to couple the beam finite element models and point-mass
elements (e.g., representation of the nacelle) to a CFD mesh.  We will leverage
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
differential-algebraic equations (DAEs) in the time domain.  We will build on experiences gained with OpenFAST and the nonlinear-beam-dynamics module, BeamDyn.

For time integration of the index-3 DAEs we will use the generalized-alpha
algorithm now established in BeamDyn which allows user-controlled numerical
damping.  See Arnold and Brüls (2007) for details.

The primary model for turbine blades and tower, which are slender and flexible
structures, will be nonlinear finite elements based on geometrically exact beam
theory (GEBT), which includes include bend-twist coupling. Spatial
discretization will be based on Legendre spectral finite elements (LSFEs; like
those in BeamDyn), but with the explicit ability to use the subset of 2-node
finite elements (a special case of LSFEs). 

The choice of rotation representation is being determined.  BeamDyn relies on
Wiener-Milenkovic vectorial rotation parameterization (three rotational degrees
of freedom), which has singularities. The team is currently investigating the
use of quanternions, which do not have singularities but require four rotational
degrees of freedom.

## Verification and validation cases

*Verification cases:*
- rigid-body dynamics; three-dimensional pendulum
- cantilever-beam nonlinear static roll up 

*Validation cases:*
- Princeton beam experiment
- Twisted and curved beams where the benchmarks are highly resolved
  solid-element Ansys models
- IEA 15-megawatt where benchmark is a highly resolved shell-element Ansys
  model

## Target baseline turbines

- NREL 5-megawatt reference turbine
- IEA 15-megawatt reference turbine

## High-level development timeline

CY23 Q2: The OpenTurbine team will implement a rigid-body dynamics solver following the concepts described above, i.e., DAE-3 coupling, quaternion rotation representation, and a generalized-alpha time integrator. This proof-of-concept implementation will in the main OpenTurbine repository and will inform next steps in OpenTurbine development.

CY23 Q3: Implement a general GEBT-based beam element that is appropriate for constrained multi-body simulations of a wind turbine. Enable variable order finite elements and user-defined material property definition (appropriate for modern turbine blades).  Demonstrate performance for a dynamic cantilever beam problem and compare against BeamDyn.

CY24 Q1: Demonstrate a wind turbine rotor simulation under prescribed loading and include code verification results and automated testing results. Include control system (e.g., ROSCO) and pitch control of blades. Compare simulation time against an equivalent model simulated with OpenFAST

CY24 Q3: Demonstrate a rotor simulation with fluid-structure interaction and a pitch control system. Fluid will be represented in two ways. First, through a simple BEMT solver and, second, where the blades are represented as actuator lines in the fluid domain (solved with the ExaWind CFD code). 

CY25 Q1: Release a robust, well documented, well tested version of OpenTurbine for land-based simulations. Demonstrate whole turbine simulation (tower, nacelle, drivetrain) with fluid-structure-interaction coupling to ExaWind.

