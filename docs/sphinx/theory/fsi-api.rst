.. _`sec:fsi`:

Coupling for fluid-structure-interaction
----------------------------------------

OpenTurbine was designed as a flexible multibody dynamics solver to be
coupled with external modules for fluid forcing at various
fluid-dynamics-model fidelity levels. At the lowest level, OpenTurbine
is coupled to a blade-element or blade-element-momentum-theory (BEMT)
solver like AeroDyn. At mid-fidelity, OpenTurbine is coupled to a CFD
solver, such as AMR-Wind, wherein blades are represented in the fluid as
actuator lines and forces are calculated through BE/BEMT. At the highest
fidelity, OpenTurbine is directly coupled to a geometry-resolved fluid
mesh in a solver like Nalu-Wind.

In all of these coupling approaches, the common thread is that forces
and moments are passed to OpenTurbine and OpenTurbine provides position
and velocity of the structure. For the preliminary development of
OpenTurbine and the coupling API, we assume that the fluid solver will
provide **point** force and moments that are appropriately distributed
to the nodes in a manner consistent with the beam basis functions.

In the following we describe the fluid-structure coupling between a
single **member** of OpenTurbine and a corresponding fluid model. As
discussed above, a member could be a beam, rigid body, or a massless
6-DOF point. Each member can be mapped to one or more fluid models. For
example, in a geometry-resolved CFD model, the CFD domain surrounding a
beam will often be decomposed for parallel computation, each domain tied
to a computational MPI rank.

We focus on coupling between a member with :math:`P` nodes, where
:math:`P=1` for body represented by a point, e.g. a rigid body or a
massless 6-DOE node, and a fluid model whose motion is tied to structure
motion. The interface should be such that only the following nodal data
(for :math:`P` nodes) is transferred after initialization:

.. math::

   \begin{aligned}
   \underline{f}_i \in \mathbb{R}^3, \,
   \underline{m}_i \in \mathbb{R}^3, \,
   \underline{q}_i \in \mathbb{R}^7,\,
   \underline{\dot{q}}_i \in \mathbb{R}^6, \qquad i\in \{1,\ldots,P\}
   \end{aligned}

where

.. math::

   \begin{aligned}
   \underline{q}_i = 
   \begin{bmatrix} \underline{u}_i \\
   \underline{\widehat{q}}_i
   \end{bmatrix} \qquad
   \underline{\dot{q}}_i = 
   \begin{bmatrix} \underline{\dot{u}}_i  \\
   \underline{\omega}_i
   \end{bmatrix} 
   \end{aligned}

and where :math:`\underline{u}_i \in \mathbb{R}^3` is displacement,
:math:`\widehat{q}_i \in \mathbb{R}^4` is relative rotation in
quaternions, and :math:`\underline{\omega}_i \in \mathbb{R}^3` is
angular velocity. In this approach, either within the fluid solver or
within an interface layer, the following data must be calculated at
initialization and be accessible to the fluid solver or interface layer:

- :math:`\underline{x}^0_\ell \in\mathbb{R}^3\,,\, \widehat{q}^0_\ell \in\mathbb{R}^4\,,\ell \in \{1, \ldots, P\}`,
  structure nodal locations and orientations (quaternions) in reference
  configuration

- :math:`n^\mathrm{motion}` and :math:`n^\mathrm{force}`, which are the
  number of fluid nodes tied to structure motion and forcing,
  respectively. For BE/BEMT,
  :math:`n^\mathrm{motion} = n^\mathrm{force}`.

- :math:`\underline{x}^{\mathrm{motion},0}_j\in\mathbb{R}^3\,,\, j \in \{1, \ldots, n^\mathrm{motion}\}`,
  fluid nodal locations in reference configuration (user provided; or
  calculated based on aerodynamic section data for BE/BEMT)

- :math:`\underline{x}^{\mathrm{force},0}_j\in\mathbb{R}^3\,,\, j \in \{1, \ldots, n^\mathrm{force}\}`,
  fluid nodal locations in reference configuration (user provided; or
  calculated based on aerodynamic section data for BE/BEMT)

- :math:`\xi^{\mathrm{motion},\mathrm{map}}_j\in[-1,1]\,,\, j \in \{1, \ldots, n^\mathrm{motion}\}`
  nearest location on the beam reference line for
  :math:`\underline{x}^{\mathrm{motion},0}_j`; provided by user for
  BE/BEMT, but must be calculated at initialization for CFD coupling

- :math:`\xi^{\mathrm{force},\mathrm{map}}_j\in[-1,1]\,,\, j \in \{1, \ldots, n^\mathrm{force}\}`
  nearest location on the beam reference line for
  :math:`\underline{x}^{\mathrm{force},0}_j`; provided by user for
  BE/BEMT, but must be calculated at initialization for CFD coupling

- :math:`\phi_\ell\left( \xi^{\mathrm{force},\mathrm{map}}_j\right)\,,\,
  \, \ell \in \{1, \ldots, P \},\,
  \, j \in \{1, \ldots, n^\mathrm{force} \}`, basis functions evaluated
  at the mapped nearest-neighbor location for each of the fluid nodes
  providing a force/moment value

- :math:`\phi_\ell \left(\xi^{\mathrm{motion},\mathrm{map}}_j\right)\,,
  \, \ell \in \{1, \ldots, P \}, 
  \, j \in \{1, \ldots, n^\mathrm{motion} \}`

- :math:`\underline{x}_j^{\mathrm{force},\mathrm{map},0}\in\mathbb{R}^3`
  and
  :math:`\widehat{q}_j^{\mathrm{force},\mathrm{map},0}\in\mathbb{R}^4`,
  :math:`j \in \{1,\ldots,n^\mathrm{force}\}`; calculated at
  initialization

- :math:`\underline{x}_j^{\mathrm{motion},\mathrm{map},0}\in\mathbb{R}^3`
  and
  :math:`\widehat{q}_j^{\mathrm{motion},\mathrm{map},0}\in\mathbb{R}^4`,
  :math:`j \in \{1,\ldots,n^\mathrm{motion}\}` ; calculated at
  initialization

- :math:`\underline{x}_j^{\mathrm{force-con}}\in\mathbb{R}^3,\, j \in \{1,\ldots,n^\mathrm{force}\}`,
  reference vectors between force-providing fluid nodes and nearest
  neighbor on the element; calculated at initialization

- :math:`\underline{x}_j^{\mathrm{motion-con}}\in\mathbb{R}^3,\, j \in \{1,\ldots,n^\mathrm{motion}\}`,
  reference vectors between fluid nodes whose motion is driven by the
  structure and nearest neighbor on the structure element; calculated at
  initialization

Note that while this formulation is generalized for a single beam
element with :math:`P` nodes, it can be applied to a single-DOF rigid
body, in which case :math:`P=1`, or expanded to multiple beam elements.

Initialization
~~~~~~~~~~~~~~

We have set of :math:`n^\mathrm{force}` points in the fluid domain that
provide point forces/moments to the structure, and a set of
:math:`n^\mathrm{motion}` points in the fluid domain whose motion is
defined by the structure. For BE/BEMT coupling, typically
:math:`n^\mathrm{force}=n^\mathrm{motion}` and
:math:`\xi^{\mathrm{force},\mathrm{map}}_j = \xi^{\mathrm{motion},\mathrm{map}}_j`
and their definition is given as an input. For more general CFD
coupling, typically :math:`n^\mathrm{force} \ll n^\mathrm{motion}`. Each
of these force and motion fluid points must be mapped and tied to a
unique location on its associated structural member,
:math:`\underline{x}_j^{\mathrm{force,map},\mathrm{0}}` and
:math:`\underline{x}_j^{\mathrm{motion,map},\mathrm{0}}`, respectively.
For a point member,
:math:`\underline{x}_j^{\mathrm{force,map},\mathrm{0}} = \underline{x}_j^0`
and
:math:`\underline{x}_j^{\mathrm{motion,map},\mathrm{0}}=\underline{x}_j^0`.

For a beam member coupled to geometry-resolved CFD, our coupling is
based on minimum distance to the the finite-element represenation of the
beam reference line. To that end, we find
:math:`\xi^{\mathrm{force},\mathrm{map}}_l \in [-1,1]` such that the
distance squared,

.. math::

   \begin{aligned}
   d_i^2 = \left(\underline{x}^{\mathrm{force},0}_i 
   - \sum_{\ell=1}^P \phi_\ell(\xi) \underline{x}^0_\ell\right)^2
   \end{aligned}

is minimized for all :math:`i \in \{1, \ldots, n^\mathrm{force} \}` and
find :math:`\xi^{\mathrm{motion},\mathrm{map}}_j \in [-1,1]`, such that
the distance squared,

.. math::

   \begin{aligned}
   d_j^2 = \left(\underline{x}^{\mathrm{motion},0}_j 
   - \sum_{\ell=1}^P \phi_\ell(\xi) \underline{x}_\ell^0\right)^2
   \end{aligned}

is minimized for all :math:`j \in \{1, \ldots, n^\mathrm{motion} \}`.
**The Jenkins–Traub algorithm, RPOLY, should be considered for these
root solving problems.** The locations of those mapped reference points
in the inertial coordinate system are given by

.. math::

   \begin{aligned}
   \underline{x}^{\mathrm{force},\mathrm{map},\mathrm{0}}_i = 
   \sum_{\ell=1}^{P} \phi_\ell(\xi^{\mathrm{force},\mathrm{map}}_i) \underline{x}^\mathrm{0}_\ell, \qquad i \in \{ 1, \ldots, n^\mathrm{force} \}
   \end{aligned}

.. math::

   \begin{aligned}
   \underline{x}^{\mathrm{motion},\mathrm{map},\mathrm{0}}_j = 
   \sum_{\ell=1}^{P} \phi_\ell(\xi^{\mathrm{motion},\mathrm{map}}_j) \underline{x}^\mathrm{0}_\ell, \qquad j \in \{ 1, \ldots, n^\mathrm{motion} \}
   \end{aligned}

.. math::

   \begin{aligned}
   \widehat{q}^{\mathrm{force,map,0}}_i &= \frac{ \sum_{\ell=1}^{P} \phi_\ell\left(\xi_i^{\mathrm{force,map}} \right) \widehat{q}^0_\ell}
   {\left \Vert \sum_{\ell=1}^{P} \phi_\ell\left(\xi_i^\mathrm{force,map} \right) \widehat{q}^0_\ell \right \Vert} \\
   \widehat{q}^{\mathrm{motion,map,0}}_j &= \frac{ \sum_{\ell=1}^{P} \phi_\ell\left(\xi_j^{\mathrm{motion,map}} \right) \widehat{q}^0_\ell}
   {\left \Vert \sum_{\ell=1}^{P} \phi_\ell\left(\xi_j^\mathrm{motion,map} \right) \widehat{q}^0_\ell \right \Vert} 
   \end{aligned}

where :math:`P` is the number of nodes in the structural element, and
:math:`\underline{x}^\mathrm{0}_\ell` and
:math:`\widehat{q}^0_\ell`\ are the reference locations and orientations
(represented as quaternions), respectively of the structural nodes in
the inertial coordinate system. For a beam coupled to a BE/BEMT solver,
:math:`\xi_j^\mathrm{motion,map} = \xi_j^\mathrm{force,map}` and those
are provided by the user. The vectors connecting these points are given
by

.. math::
   \begin{aligned}
   \underline{x}^\mathrm{force-con}_i &= -\underline{x}^{\mathrm{force},\mathrm{0}}_i + \underline{x}_i^{\mathrm{force},\mathrm{map}\mathrm{0}},  \qquad i \in \{ 1, \ldots, n^\mathrm{force} \} \\
   \underline{x}^\mathrm{motion-con}_j &= \underline{x}_j^{\mathrm{motion},\mathrm{0}} - \underline{x}^{\mathrm{motion},\mathrm{map},\mathrm{0}}_j, \qquad j \in \{ 1, \ldots, n^\mathrm{motion} \}
   \end{aligned}

blah

.. figure:: images/fsi-map.png
   :width: 50.0%

   blah

Motion transfer: Structure to fluid nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As the first step, generalized displacements and velocities are
calculated at the mapped locations on the structure:

.. math::

   \begin{aligned}
   \underline{q}_j^{\mathrm{motion},\mathrm{map}} = 
   \begin{bmatrix} \underline{u}_j^{\mathrm{motion},\mathrm{map}} \\
   \widehat{q}_j^{\mathrm{motion},\mathrm{map}}
   \end{bmatrix} \qquad
   \underline{\dot{q}}_j^{\mathrm{motion},\mathrm{map}}
   \begin{bmatrix} \underline{\dot{u}}_j^{\mathrm{motion},\mathrm{map}} \\
   \underline{\omega}_j^{\mathrm{motion},\mathrm{map}}
   \end{bmatrix}, 
   \qquad j \in \{ 1, \ldots, n^\mathrm{motion} \}
   \end{aligned}

where

.. math::

   \begin{aligned}
   \underline{u}_j^{\mathrm{motion},\mathrm{map}} = \sum_{\ell=1}^P \phi_\ell \left(\xi_j^{\mathrm{motion},\mathrm{map}} \right) \underline{u}_\ell \\
   \widehat{q}^{\mathrm{motion},\mathrm{map}}_j = \frac{ \sum_{\ell=1}^{P} \phi_\ell\left(\xi_j^{\mathrm{motion},\mathrm{map}} \right) \widehat{q}_\ell} 
   {|| \sum_{\ell=1}^{P} \phi_\ell\left(\xi_j^{\mathrm{motion},\mathrm{map}} \right) \widehat{q}_\ell ||} \\
   \underline{\dot{q}}_j^{\mathrm{motion},\mathrm{map}} = \sum_{\ell=1}^P \phi_i \left(\xi_j^{\mathrm{motion},\mathrm{map}} \right) \underline{\dot{q}}_\ell
   \end{aligned}

The current position of the fluid nodes (in global/inertial coordinates)
is

.. math::

   \begin{aligned}
   \underline{x}_j^\mathrm{fl} = 
   \underline{x}_j^{\mathrm{motion},\mathrm{0}} 
   + \underline{u}_j^{\mathrm{motion},\mathrm{map}} + 
   \left[ \underline{\underline{R}}(\widehat{q}_j^{\mathrm{motion},\mathrm{map}}) - \underline{\underline{I}} \right] \underline{x}^\mathrm{motion-con}_j, 
   \qquad j \in \{ 1, \ldots, n^\mathrm{motion} \}
   \end{aligned}

and the current velocity of the fluid nodes is

.. math::

   \begin{aligned}
   \dot{\underline{u}}_j^\mathrm{fl} = 
   \dot{\underline{u}}_j^{\mathrm{motion},\mathrm{map}} 
   + \underline{\omega}^{\mathrm{motion},\mathrm{map}}_j \times \left[\underline{\underline{R}}(\underline{\widehat{q}}_j^{\mathrm{motion},\mathrm{map}})\underline{x}^\mathrm{motion-con}_j\right],\,
   \qquad j \in \{ 1, \ldots, n^\mathrm{motion} \}
   \end{aligned}

These are passed to the fluid solver.

Force and Moment transfer: Fluid to structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have a set of :math:`n^\mathrm{force}` forces and moments,
:math:`\underline{f}^\mathrm{force}_i` and
:math:`\underline{m}^\mathrm{force}_i`, with reference locations
:math:`\underline{x}_i^{\mathrm{force},\mathrm{0}}`. Note that in CFD
coupling, the applied moments will be zero.

We need the orientations:

.. math::

   \begin{aligned}
   \widehat{q}^{\mathrm{force},\mathrm{map}}_j = \frac{ \sum_{\ell=1}^{P} \phi_\ell\left(\xi_j^{\mathrm{force},\mathrm{map}} \right) \widehat{q}_\ell}
   {|| \sum_{\ell=1}^{P} \phi_\ell\left(\xi_j^{\mathrm{force},\mathrm{map}} \right) \widehat{q}_\ell ||}
   \,, \qquad j \in \{ 1, \ldots, n^\mathrm{force} \}  
   \end{aligned}

Nodal forces (at :math:`P` nodes) are

.. math::

   \begin{aligned}
   \underline{f}_\ell = \sum_{j=1}^{n^\mathrm{force}} \phi_\ell(\xi^{\mathrm{force},\mathrm{map}}_j) \underline{f}^\mathrm{force}_j, \qquad \ell \in \{ 1, \ldots, P \}
   \label{eq:force}
   \end{aligned}

Nodal moments (at :math:`P` nodes) are

.. math::

   \begin{aligned}
   \underline{m}_\ell = \sum_{j=1}^{n^\mathrm{force}} \phi_\ell(\xi^{\mathrm{force},\mathrm{map}}_j) \left[\underline{f}^\mathrm{force}_j \times \left( \underline{\underline{R}}(\widehat{q}^{\mathrm{force},\mathrm{map}}_j) \underline{x}^\mathrm{force-con}_j\right) + \underline{m}^\mathrm{force}_j\right], \qquad \ell \in \{ 1, \ldots, P \}
   \label{eq:moment}
   \end{aligned}
