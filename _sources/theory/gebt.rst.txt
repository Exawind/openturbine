.. _sec-gebt:

Geometrically exact beam theory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenTurbine beam finite elements are based on geometrically exact beam
theory (GEBT) [@Reissner:1973 @Simo:1985; @Simo-VuQuoc:1986]. Our
formulation largely follows that described in [@Bauchau:2011] and as
implemented in Dymore [@Dymore:2013] and the BeamDyn [@Wang-etal:2017]
module of OpenFAST. A key difference is that OpenTurbine uses
quaternions to store and track rotations rather than Wiener-Milenkovic
parameters, thereby removing the challenges associated rescaling
operations to avoid singularities.

In our description of GEBT, we focus on a single beam and assume it is
defined in its reference configuration by a smooth curve, called the
beam reference line, :math:`\underline{x}^0(s)\in\mathbb{R}^3`,
:math:`\underline{\underline{R}}^0(s) \in \mathrm{SO(3)}` parameterized
by :math:`s \in [0, L]`, where :math:`L` is the arclength of the
reference line. We denote reference position as

.. math::

   \begin{aligned}
    \underline{q}^\mathrm{0} = \begin{bmatrix}
     \underline{x}^\mathrm{0} \\
     \underline{\underline{R}}^\mathrm{0} \\
    \end{bmatrix}
   \end{aligned}

where :math:`\underline{q}^0(s) \in \mathbb{R}^3\times \mathrm{SO(3)}`.
Generalized displacements are denoted

.. math::

   \begin{aligned}
    \underline{q} = \begin{bmatrix}
     \underline{u} \\
     \underline{\underline{R}}\\
    \end{bmatrix}
   \end{aligned}

where :math:`\underline{q}(s,t) \in \mathbb{R}^3\times \mathrm{SO(3)}`,
:math:`\underline{\underline{R}}(s,t)\in \mathrm{SO(3)}` is the
relative-rotation tensor, and :math:`\underline{u}(s,t)\in \mathbb{R}^3`
is the displacement. The beam current (deformed) configuration at time
:math:`t` is given by

.. math::

   \begin{aligned}
   \underline{x}^\mathrm{c} &= \underline{x}^0 + \underline{u}\\
   \underline{\underline{R}}^\mathrm{c} &= \underline{\underline{R}}\,\underline{\underline{R}}^0 
   \end{aligned}

The GEBT equations are motion are given by

.. math::

   \begin{aligned}
   \dot{\underline{h}} - \underline{\mathcal{F}}^\prime &= \underline{f}\\
   \dot{\underline{g}} + \widetilde{u}\, \underline{h} - \underline{\mathcal{M}}^\prime - \left(\widetilde{x}^{0\prime} + \widetilde{u}^\prime \right) \underline{\mathcal{F}}&= \underline{m}
   \end{aligned}

where :math:`\underline{h}(s,t),\underline{g}(s,t) \in \mathbb{R}^3` are
linear and angular momenta, respectively, resolved in the inertial
coordinate system,
:math:`\underline{\mathcal{F}}(s,t),\underline{\mathcal{M}}(s,t) \in \mathbb{R}^3`
are section force and moments, respectively,
:math:`\underline{f}(s,t),\underline{m}(s,t)\in \mathbb{R}^3` are
externally applied forces and moments, respectively, a prime denotes a
spatial derivate along the beam reference line, and an overdot denotes a
time derivative. The constitutive equations are given by

.. math::

   \begin{aligned}
   \begin{bmatrix}  \underline{h} \\ \underline{g} \end{bmatrix} 
   &= \underline{\underline{M}} 
   \begin{bmatrix}  \dot{\underline{u}} \\ \underline{\omega} \end{bmatrix} \\
   \begin{bmatrix}  \underline{\mathcal{F}} \\ \underline{\mathcal{M}} \end{bmatrix} 
   &= \underline{\underline{C}} 
   \begin{bmatrix}  \underline{\epsilon} \\ \underline{\kappa} \end{bmatrix}
    \label{eq:constitutive}
   \end{aligned}

where
:math:`\underline{\underline{M}}(s,t), \underline{\underline{C}}(s,t) \in \mathbb{R}^{6\times6}`
are the sectional mass and stiffness matrices in inertial coordinates,
and
:math:`\underline{\epsilon}(s,t),\underline{\kappa}(s,t) \in \mathbb{R}^3`
are the sectional strain and curvature defined in inertial coordinates,
which are defined as

.. math:: \underline{\epsilon} = \underline{x}^{0\prime}+\underline{u}^\prime-\left(\underline{\underline{R}}\,\underline{\underline{R}}^0\right) \hat{i}_1 
   :label: strain 

.. math::
    \underline{\kappa} = \mathrm{axial}\left({ \underline{\underline{R}}^\prime \underline{\underline{R}} }\right)

respectively. In the reference configuration, i.e.,
:math:`\underline{\underline{R}}=\underline{\underline{I}}` and
:math:`\underline{u}=0`, zero strain requires that

.. math:: \underline{x}^{0\prime} = \underline{\underline{R}}^0 \hat{i}_1.
   :label: zerostrain

In our finite-element implementation of
Eq. :eq:`strain`, :math:`\underline{x}^{0\prime}` will
be interpolated from nodal values and
:math:`\underline{\underline{R}}^0` will be constructed from quaternions
interpolated from nodal values. However, in general, at non-node
locations, Eq. :eq:`zerostrain` is not guaranteed
in the reference configuation due to interpolation differences. Hence,
we use the equivalent definition of strain, which guarantees zero strain
in the reference configuration at nodal and interpolated locations:

.. math::

   \begin{aligned}
    \underline{\epsilon} &= \underline{x}^{0\prime}+\underline{u}^\prime-\underline{\underline{R}}\,\underline{x}^{0\prime} 
   \label{eq:newstrain}
   \end{aligned}

Sectional stiffness and mass matrices are typically defined in material
coordinates (denoted by a :math:`*` superscript), from which the
inertial coordinate section matrices can be calculated as

.. math::

   \begin{aligned}
   \underline{\underline{C}} = \underline{\underline{\mathcal{RR}^0}}\, \underline{\underline{C}}^*\, \underline{\underline{\mathcal{RR}^0}}^T\\
   \underline{\underline{M}} = \underline{\underline{\mathcal{RR}^0}}\, \underline{\underline{M}}^*\, \underline{\underline{\mathcal{RR}^0}}^T\\
   \end{aligned}

where
:math:`\underline{\underline{C}}^*(s), \underline{\underline{M}}^*(s) \in \mathbb{R}^{6\times6}`.
As described below, sectional mass and stiffness matrices in material
coordinates, and twist about the reference line, are user-defined
quantities in :math:`s\in[0,L]`.

The governing equations can be written as a residual expression in
strong form as

.. math::
   \underline{\mathcal{R}} = 
   \underline{\mathcal{F}}^I -
   \underline{\mathcal{F}}^{C\prime} +
   \underline{\mathcal{F}}^D -
   \underline{\mathcal{F}}^\mathrm{ext} 
   :label: stronggoverning

where each term is in :math:`\mathbb{R}^6`;
:math:`\underline{\underline{\mathcal{F}}}^I(s,t)` is the interial
force, :math:`\underline{\underline{\mathcal{F}}}^C(s,t)` and
:math:`\underline{\underline{\mathcal{F}}}^D(s,t)` are elastic forces,
and :math:`\underline{\underline{\mathcal{F}}}^\mathrm{ext}` are the
external forces and moments. The inertial force in the inertial frame is

.. math::

   \begin{aligned}
   \underline{\mathcal{F}}^I =  
   \begin{bmatrix}
   \dot{\underline{h}} \\ \dot{\underline{g}} + \dot{\widetilde{u}} \underline{g}
   \end{bmatrix}
   = \begin{bmatrix}
   m \ddot{\underline{u}} +
   \left( \dot{\widetilde{\omega}}+ \widetilde{\omega} \widetilde{\omega} \right) m \underline{\eta}\\
   m \widetilde{\eta} \ddot{\underline{u}} + \underline{\underline{\rho}} \dot{\underline{\omega}}
    + \widetilde{\omega} \underline{\underline{\rho}} \underline{\omega}
   \end{bmatrix}
   = \underline{\underline{M}}(\underline{q}) \dot{\underline{v}} + \begin{bmatrix} 
    m \widetilde{\omega}\widetilde{\omega} \underline{\eta} \\
   \widetilde{\omega} \underline{\underline{\rho}} \underline{\omega} 
   \end{bmatrix}
   \end{aligned}

where :math:`m`, :math:`\underline{\eta}`, and
:math:`\underline{\underline{\rho}}` are readily extracted from the
section mass matrix in inertial coordinates as

.. math::

   \begin{aligned}
   \underline{\underline{M}} = 
   \begin{bmatrix}
   m \underline{\underline{I}}_3 & m \tilde{\eta}^T\\
   m \tilde{\eta} & \underline{\underline{\rho}}
   \end{bmatrix}
   \end{aligned}

The elastic forces are

.. math::

   \begin{aligned}
   \underline{\mathcal{F}}^C &= \underline{\underline{C}}\, \begin{bmatrix} \underline{\epsilon} \\ \underline{\kappa} \end{bmatrix}\\
   \underline{\mathcal{F}}^D &=
   \begin{bmatrix} \underline{0} \\ 
   \left(\tilde{x}'^0+\tilde{u}'\right)^T \left( \underline{\underline{C}}_{11} \underline{\epsilon} 
   + \underline{\underline{C}}_{12} \underline{\kappa}\right)  \end{bmatrix}
   \end{aligned}

where
:math:`\underline{\underline{C}}_{11},\underline{\underline{C}}_{12}\in \mathbb{R}^{3\times 3}`
are the submatrices of the full sectional stiffness matrix in intertial
coordinates, i.e.,

.. math::

   \begin{aligned}
   \underline{\underline{C}} = \begin{bmatrix}
   \underline{\underline{{C}}}_{11} & \underline{\underline{{C}}}_{12} \\
   \underline{\underline{{C}}}_{21} & \underline{\underline{{C}}}_{22} \end{bmatrix}
   \end{aligned}

We describe the variation of the residual,
Eq. :eq:`residual1`, in parts. Variation of the
inerial forces can be written

.. math::

   \begin{aligned}
   \delta \underline{\mathcal{F}}^I =
   \underline{\underline{\mathcal{M}}}^I \delta \ddot{\underline{q}}
   + \underline{\underline{\mathcal{G}}}^I \delta \dot{\underline{q}}
   + \underline{\underline{\mathcal{K}}}^I \delta \underline{q}
   \end{aligned}

where

.. math::

   \begin{aligned}
   \underline{\underline{\mathcal{G}}} =
   \begin{bmatrix}
   \underline{\underline{0}} & \widetilde{ \widetilde{\omega} m \underline{\eta} }^T
            + \widetilde{\omega} m \widetilde{\eta}^T\\
   \underline{\underline{0}} & \widetilde{\omega} \underline{\underline{\rho}} - \widetilde{\underline{\underline{\rho}} \underline{\omega}}
   \end{bmatrix}
   \end{aligned}

.. math::

   \begin{aligned}
   \underline{\underline{\mathcal{K}}} =
   \begin{bmatrix}
   \underline{\underline{0}} & \left( \dot{\widetilde{\omega}} + \tilde{\omega}\tilde{\omega}
           \right) m \widetilde{\eta}^T\\
   \underline{\underline{0}} & \ddot{\widetilde{u}} m \widetilde{\eta}
            + \left(\underline{\underline{\rho}}\dot{\widetilde{\omega}}
                    -\widetilde{\underline{\underline{\rho}} \dot{\underline{\omega}}} \right)
            + \widetilde{\omega} \left( \underline{\underline{\rho}} \widetilde{\omega}
            - \widetilde{ \underline{\underline{\rho}}\underline{\omega}} \right)
   \end{bmatrix}
   \end{aligned}

Variation of the elastic forces are as follows:

.. math::

   \begin{aligned}
   \delta \underline{\mathcal{F}}^C =
   \underline{\underline{\mathcal{C}}} \delta \underline{q}' + \underline{\underline{\mathcal{O}}} \delta \underline{q}
   \end{aligned}

.. math::

   \begin{aligned}
   \underline{\underline{\mathcal{O}}} =
   \begin{bmatrix}
   \underline{\underline{0}} &  -\widetilde{N} + \underline{\underline{\mathcal{C}}}_{11}\left(  \tilde{x}^{0 \prime}+ \tilde{u}' \right)   \\
   \underline{\underline{0}} &  -\widetilde{M} + \underline{\underline{\mathcal{C}}}_{21}\left( \tilde{x}^{0 \prime} + \tilde{u}' \right)
   \end{bmatrix}
   \end{aligned}

.. math::

   \begin{aligned}
   \delta \underline{\mathcal{F}}^D =
   \underline{\underline{\mathcal{P}}} \delta \underline{q}' + \underline{\underline{\mathcal{Q}}} \delta \underline{q}
   \end{aligned}

.. math::

   \begin{aligned}
   \underline{\underline{\mathcal{P}}} =
   \begin{bmatrix}
   \underline{\underline{0}} & \underline{\underline{0}} \\
    \widetilde{N} + \left(  \tilde{x}^{0 \prime}+ \tilde{u}' \right)^T
   \underline{\underline{\mathcal{C}}}_{11} &
   \left( \tilde{x}^{0 \prime} + \tilde{u}' \right)^T
   \underline{\underline{\mathcal{C}}}_{12}
   \end{bmatrix}
   \end{aligned}

.. math::

   \begin{aligned}
   \underline{\underline{\mathcal{Q}}} =
   \begin{bmatrix}
   \underline{\underline{0}} & \underline{\underline{0}} \\
    \underline{\underline{0}} &
   \left( \tilde{x}^{0 \prime} + \tilde{u}' \right)^T
   \left[-\widetilde{N} + \underline{\underline{C}}_{11} \left( \tilde{x}^{0 \prime} + \tilde{u}' \right) \right]
   \end{bmatrix}
   \end{aligned}

.. container:: references csl-bib-body hanging-indent
   :name: refs

   .. container:: csl-entry
      :name: ref-Bauchau:2011

      Bauchau, O. A. 2011. *Flexible Multibody Dynamics*. Springer.

   .. container:: csl-entry
      :name: ref-Dymore:2013

      ———. 2013. “Dymore User’s Manual.”

   .. container:: csl-entry
      :name: ref-Reissner:1973

      Reissner, E. 1973. “On One-Dimensional Large-Displacement
      Finite-Strain Beam Theory.” *Studies in Applied Mathematics LII*,
      87–95.

   .. container:: csl-entry
      :name: ref-Simo:1985

      Simo, J. C. 1985. “A Finite Strain Beam Formulation. The
      Three-Dimensional Dynamic Problem. Part I.” *Computer Methods in
      Applied Mechanics and Engineering* 49: 55–70.

   .. container:: csl-entry
      :name: ref-Simo-VuQuoc:1986

      Simo, J. C., and L. Vu-Quoc. 1986. “A Three-Dimensional
      Finite-Strain Rod Model. Part II.” *Computer Methods in Applied
      Mechanics and Engineering* 58: 79–116.

   .. container:: csl-entry
      :name: ref-Wang-etal:2017

      Wang, Q., M. A. Sprague, J. Jonkman, N. Johnson, and B. Jonkman.
      2017. “BeamDyn: A High-Fidelity Wind Turbine Blade Solver in the
      FAST Modular Framework.” *Wind Energy*.
      https://doi.org/10.1002/we.2101.
