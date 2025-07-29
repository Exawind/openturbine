Legendre spectral finite elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The GEBT equations are discretized with Legendre spectral finite
elements (LSFEs), which are defined by :math:`P` nodes and
:math:`(P-1)`\ th order Lagrangian-interpolant basis functions. Nodes
are located at the :math:`P` Gauss-Legendre-Lobatto (GLL) points (in the
element reference domain), :math:`\xi_i` for
:math:`i\in\{1,2, \ldots,P\}`, which are the solutions to the equation

.. math::

   \begin{aligned}
    \left( 1-\xi^2\right) \frac{\partial L_{P-1}}{\partial \xi} = 0
   \end{aligned}

for :math:`\xi \in[-1,1]`, where :math:`L_{P-1}(\xi)` is the Legendre
polynomial of degree :math:`(P-1)` [@Deville-etal:2002]. The beam
reference line is approximated as

.. math::

   \begin{aligned}
   \underline{x}^{0,h}(\xi) =  \sum_{\ell=1}^{P} \phi_\ell(\xi) \underline{x}^0_\ell
   \end{aligned}

where :math:`\phi_\ell(\xi)` is the Lagrangian-interpolant of the
:math:`\ell^{th}` node. For LSFEs, those can be written as

.. math::

   \begin{aligned}
   \phi_\ell(\xi) = \frac{-1}{P (P-1)} 
   \frac{\left(1-\xi^2\right) \frac{\partial L_{P-1}}{\partial \xi}}{(\xi - \xi_\ell) L_{P-1}(\xi_\ell)}
   \end{aligned}

for :math:`\xi \in [-1,1]`. The weak form of the residual of the
governing equations Eq. :eq:`stronggoverning`,
for the :math:`i^\mathrm{th}`-node, can be written (after integration by
parts) as

.. math::
   \underline{R}_i = 
   \int_{-1}^{1} \phi_i \underline{\mathcal{R}}\, J d\xi = 
   \int_{-1}^{1} \left(
   \phi_i \underline{\mathcal{F}}^I +
   \frac{\partial \phi_i}{\partial \xi} \underline{\mathcal{F}}^{C} J^{-1} +
   \phi_i \underline{\mathcal{F}}^D  - 
   \phi_i\underline{\mathcal{F}}^\mathrm{ext}\right)
    J d\xi
   :label: weakresidual

:math:`\underline{R}\in\mathbb{R}^6`, for all
:math:`i\in\{1,\ldots,P\}`, where we have mapped the domain
:math:`s\in[0,L]` to :math:`\xi \in[-1,1]`,
:math:`J(\xi) \in \mathbb{R}` is the Jacobian of the mapping,

.. math::

   \begin{aligned}
   J(\xi) = \sqrt{\frac{\partial \underline{x}^{0,h}}{\partial \xi}^T \frac{\partial \underline{x}^{0,h}}{\partial \xi} }
   \end{aligned}

We remark that standard linear (i.e., 2-node) and quadratic (i.e.,
3-node) elements are a subset of LSFEs.

As described above, the generalized displacement, :math:`\underline{q}`
is in :math:`\mathbb{R}^3 \times \mathrm{SO(3)}`. However, rotations are
represented as quaternions for storage and interpolation, and are stored
at nodes. In this form, nodal degrees of freedom are denoted, for the
:math:`i^\mathrm{th}` node,

.. math::

   \begin{aligned}
    \underline{q}_i = \begin{bmatrix}
     \underline{u}_i \\
     \hat{q}_i \\
   \end{bmatrix}
   \end{aligned}

where :math:`\underline{u}_i \in \mathbb{R}^3` and
:math:`\hat{q}_i \in \mathbb{R}^4`. The generalized displacement in
:math:`\mathbb{R}^3\times\mathrm{SO(3)}` along the beam reference line
is then given as

.. math::

   \begin{aligned}
    \underline{q}(s,t) = \begin{bmatrix} \underline{u}^h \\ \underline{\underline{R}} \left( \widehat{q}^h \right)
   \end{bmatrix}
   \end{aligned}

where displacement is interpolated in the normal manner, i.e.,

.. math::

   \begin{aligned}
   \underline{u}^h =  \sum_{j=1}^{p} \phi_j\underline{u}_j
   \end{aligned}

but quaternion interpolation requires normaliation, i.e.,

.. math::

   \begin{aligned}
   \widehat{q}^h = \frac{ \sum_{j=1}^{p} \phi_j \hat{q}_j }
   {|| \sum_{j=1}^{p} \phi_j \hat{q}_j ||}
   \end{aligned}

For a given quaternion, the associated rotation matrix is calculated as

.. math::

   \begin{aligned}
   \underline{\underline{R}}\left(\hat{q}\right) = \underline{\underline{I}} + q \widetilde{q} + 2 \widetilde{q} \widetilde{q}
   \end{aligned}

Introducing a quadrature scheme with :math:`n^Q` points with locations
and weights, :math:`\xi_j^Q`, :math:`w_j^Q`,
:math:`j\in \{1, \ldots, n^Q\}`, respectively, the approximate form of
the residual, Eq. :eq:`weakresidual`, can be
written

.. math::

   \begin{aligned}
   \underline{R} = \begin{bmatrix}
   \underline{F}^I_1 + \underline{F}^E_1 - \underline{F}^\mathrm{ext}_1 \\
   \underline{F}^I_2 + \underline{F}^E_2 - \underline{F}^\mathrm{ext}_2 \\
   \vdots \\
   \underline{F}^I_P + \underline{F}^E_P - \underline{F}^\mathrm{ext}_P 
   \end{bmatrix}
   \end{aligned}

where :math:`\underline{R} \in \mathbb{R}^{6 P}` and

.. math::

   \begin{aligned}
   \underline{F}_i^{I} &=
   \sum_{j=0}^{n^Q}
   J(\xi^Q_j) \phi_i(\xi^Q_j) \underline{\mathcal{F}}^I(\xi^Q_j) w^Q_j\, \\
   \underline{F}_i^E &=
   \sum_{j=0}^{n^Q}
   \left[ \left .\frac{\partial \phi_i}{\partial \xi}\right |_{\xi=\xi^Q_j}
   {\underline{\mathcal{F}}^{C}}(\xi^Q_j)+ J(\xi^Q_j) \phi_i(\xi^Q_j) \underline{\mathcal{F}}^D(\xi^Q_j) \right] w^Q_j\, \\
   \underline{F}_i^{ext} &=
   \sum_{j=0}^{n^Q} \phi_i (\xi^Q_j)
   \underline{F}^{ext}(\xi^Q_j) J(\xi^Q_j) w^Q_j 
   \end{aligned}

The matrices required for the time-integration iteration matrix in
Eq. :eq:`iteration` are constructed from the
following:

.. math::

   \begin{aligned}
   \underline{\underline{M}}_{ij} =
   \sum_{k=1}^{n^Q} &
   \phi_i(\xi^Q_k) \underline{\underline{\mathcal{M}}}(\xi^Q_k) \phi_j(\xi^Q_k) J(\xi^Q_k) w^Q_k \\
   %
   \underline{\underline{G}}_{ij} =
   \sum_{k=1}^{n^Q} &
   \phi_i(\xi^Q_k) \underline{\underline{\mathcal{G}}}(\xi^Q_k) \phi_j(\xi^Q_k) J(\xi^Q_k) w^Q_k \\
   %
   \underline{\underline{K}}_{ij} =
   \sum_{k=1}^{n^Q} 
   \Big\{ & \phi_i(\xi^Q_k) \underline{\underline{\mathcal{P}}}(\xi^Q_k) \phi'_j(\xi^Q_k) +
   \phi_i(\xi^Q_k) \left[\underline{\underline{\mathcal{K}}}(\xi^Q_k)+\underline{\underline{\mathcal{Q}}}(\xi^Q_k) \right]\phi_j(\xi^Q_k) J(\xi^Q_k)+ \\
   &
   \phi'_i(\xi^Q_k) \underline{\underline{\mathcal{C}}}(\xi^Q_k) \phi'_j(\xi^Q_k) \frac{1}{J(\xi^Q_k)}+
   \phi'_i(\xi^Q_k) \underline{\underline{\mathcal{O}}}(\xi^Q_k) \phi_j(\xi^Q_k)
   \Big\} w^Q_k \\
   \end{aligned}

for all :math:`i,j \in\{1,2, \ldots, P\}` and
:math:`\underline{\underline{M}}_{ij},\underline{\underline{G}}_{ij},\underline{\underline{K}}_{ij} \in \mathbb{R}^{6 \times 6}`.
These matrices define the full matrices for a single beam LSFE. For
example, the mass matrix is assembled as

.. math::

   \underline{\underline{M}} =
   \begin{bmatrix}
   \underline{\underline{M}}_{00}&
   \underline{\underline{M}}_{01}& \ldots &
   \underline{\underline{M}}_{0P}\\
   \underline{\underline{M}}_{10} &
   \underline{\underline{M}}_{11}&
   \ldots &
   \underline{\underline{M}}_{1P}\\
   \vdots & \vdots & \vdots & \vdots \\
   \underline{\underline{M}}_{P0}&
   \underline{\underline{M}}_{P1}& \ldots &
   \underline{\underline{M}}_{PP}\\
   \end{bmatrix}

where
:math:`\underline{\underline{M}} \in \mathbb{R}^{6 (P+1) \times 6 (P+1)}`;
similarly for
:math:`\underline{\underline{G}}, \underline{\underline{K}} \in \mathbb{R}^{6 (P+1) \times 6 (P+1)}`.

.. container:: references csl-bib-body hanging-indent
   :name: refs

   .. container:: csl-entry
      :name: ref-Deville-etal:2002

      Deville, M. O., P. F. Fischer, and E. H. Mund. 2002. *High-Order
      Methods for Incompressible Fluid Flow*. Cambridge University
      Press.
