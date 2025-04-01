SO(3)-based GEBT Beam
=====================

This document's purpose is to document the equations needed for a spectral-finite element implementation of Geometrically Exact Beam Theory (GEBT), where rotations are modeled in SO(3), rather than as Wiener-Milenkovicz parameters like in BeamDyn. The derivation is based on that described in Bauchau (2011).

Useful Equations
----------------

.. math::

   \underline{\psi} = \log \left(\underline{\underline{R}}\right)

   \underline{\delta \psi} = \mathrm{axial}\left(\delta \underline{\underline{R}}\,\underline{\underline{R}}^T\right)

   \underline{\omega} = \mathrm{axial}\left(\dot{\underline{\underline{R}}}\,\underline{\underline{R}}^T\right)

   \underline{\kappa} = \mathrm{axial}\left(\underline{\underline{R}}'\,\underline{\underline{R}}^T\right)

which are the Cartesian rotation vector, the virtual rotation vector, the angular velocity (:math:`\in \mathbb{R}^3`), and the curvature (:math:`\in \mathbb{R}^3`), respectively, for rotation tensor :math:`\underline{\underline{R}} \in SO(3)`, and where a overdot denotes a time derivative and a prime denotes spatial derivative along the beam's reference line.

Also useful: For any :math:`\underline{a} \in \mathbb{R}^3` and :math:`\underline{\underline{A}} \in \mathbb{R}^{3\times 3}`,

.. math::

   \tilde{a} =
   \begin{bmatrix}
   0 & -a_3 & a_2 \\
   a_3 & 0 & -a_1 \\
   -a_2 & a_1 & 0
   \end{bmatrix}

.. math::

   \mathrm{axial}\left(\underline{\underline{A}}\right) =
   \frac{1}{2}
   \begin{bmatrix}
   a_{32}-a_{23} \\
   a_{13}-a_{31} \\
   a_{21}-a_{12}
   \end{bmatrix}

If :math:`\underline{\underline{R}}` is represented by an Euler parameter, i.e., unit quaternion, :math:`\hat{q}^T= (q_0,q_1,q_2,q_3)`,

.. math::

   \underline{\omega} = 2 F(\hat{q}) \dot{\hat{q}}

   \underline{\kappa} = 2 F(\hat{q}) \hat{q}'

where

.. math::

   F =
   \begin{bmatrix}
   -q_1 & +q_0 & -q_3 &  +q_2  \\
   -q_2 & +q_3 & +q_0 &  -q_1  \\
   -q_3 & -q_2 & +q_1 &  +q_0  \\
   \end{bmatrix}

For any :math:`\underline{a},\underline{b} \in \mathbb{R}^3`,

.. math::

   \widetilde{a}\underline{b} = -\widetilde{b}\underline{a}

   \widetilde{a}^T = -\widetilde{a}

   \widetilde{ \widetilde{a} \underline{b} } - \widetilde{a} \widetilde{b} = - \widetilde{b} \widetilde{a}

Define:

.. math::

   \underline{\underline{\mathcal{RR}^0}} =
   \begin{bmatrix}
   \underline{\underline{R}}~\underline{\underline{R}}^0 & \underline{\underline{0}} \\
   \underline{\underline{0}} & \underline{\underline{R}}~\underline{\underline{R}}^0
   \end{bmatrix}

for :math:`\underline{\underline{\mathcal{RR}^0}} \in \mathbb{R}^{6\times6}`

Sectional stiffness and mass matrices (see B, p. 688):

.. math::

   \underline{\underline{\mathcal{C}}} = \underline{\underline{\mathcal{RR}^0}}\, \underline{\underline{\mathcal{C}}}^*\, \underline{\underline{\mathcal{RR}^0}}^T

   \underline{\underline{\mathcal{M}}} = \underline{\underline{\mathcal{RR}^0}}\, \underline{\underline{\mathcal{M}}}^*\, \underline{\underline{\mathcal{RR}^0}}^T

for user-provided property-matrices :math:`\underline{\underline{\mathcal{C}}}^*, \underline{\underline{\mathcal{M}}}^* \in \mathbb{R}^{6\times6}`.

Reference beam line in :math:`0 \le \zeta  \le L`:

.. math::

   \underline{\mathcal{Q}}^0 =
   \begin{Bmatrix}
   \underline{x}^0(\zeta) \\
   \underline{\underline{R}}^0(\zeta)
   \end{Bmatrix}

where :math:`\underline{\mathcal{Q}}^0 \in G = \mathbb{R}^3 \times SO(3)`

Generalized displacements (intertal basis) of beam line in :math:`0 \le \zeta \le L`:

.. math::

   \underline{\mathcal{Q}} =
   \begin{Bmatrix}
   \underline{u}(\zeta) \\
   \underline{\underline{R}}(\zeta)
   \end{Bmatrix}

where :math:`\underline{\mathcal{Q}} \in G = \mathbb{R}^3 \times SO(3)`

Generalized velocities (intertal basis) of beam line in :math:`0 \le \zeta \le L`:

.. math::

   \underline{\mathcal{V}} =
   \begin{Bmatrix}
   \dot{\underline{u}}(\zeta) \\
   \underline{\omega}(\zeta)
   \end{Bmatrix}

Generalized velocities (material basis) of beam line in :math:`0 \le \zeta \le L`:

.. math::

   \underline{\mathcal{V}}^* = \underline{\underline{\mathcal{RR}^0}}^T \mathcal{V}

Sectional strain measure for a beam in the inertial basis (B 16.45):

.. math::

   \underline{e}=
   \begin{Bmatrix}
   \underline{\epsilon} \\ \underline{\kappa}
   \end{Bmatrix}
   = \begin{Bmatrix}
   \underline{x}{^0}'+\underline{u}'-\left(\underline{\underline{R}}~\underline{\underline{R}}^0 \right) \overline{i}_1 \\
   \mathrm{axial}\left(\underline{\underline{R}}'\underline{\underline{R}}^T\right)
   \end{Bmatrix}

Sectional strain measure for a beam in the material basis:

.. math::

   \underline{e}^*=
   \begin{Bmatrix}
   \underline{\epsilon}^* \\ \underline{\kappa}^*
   \end{Bmatrix}
   = \begin{Bmatrix}
   \left( \underline{\underline{R}}~\underline{\underline{R}}^0 \right)^T \left(\underline{x}^{0\prime}+\underline{u}'\right)-\overline{i}_1 \\
   \left( \underline{\underline{R}}~\underline{\underline{R}}^0 \right)^T \mathrm{axial}\left(\underline{\underline{R}}'\underline{\underline{R}}^T\right)
   \end{Bmatrix}

Sectional elastic forces and moments in material/inertial systems (B 16.47):

.. math::

   \begin{Bmatrix}
   \underline{N}^* \\ \underline{M}^*
   \end{Bmatrix}
   =\underline{\underline{\mathcal{C}}}^* \underline{e}^* \qquad
   \begin{Bmatrix}
   \underline{N} \\ \underline{M}
   \end{Bmatrix}
   =\underline{\underline{\mathcal{C}}}\, \underline{e}

for :math:`\underline{\underline{\mathcal{C}}}^*, \underline{\underline{\mathcal{C}}} \in \mathbb{R}^{6\times6}`, where :math:`\underline{\underline{\mathcal{C}}} = \underline{\underline{\mathcal{RR}^0}}\, \underline{\underline{\mathcal{C}}}^*\, \underline{\underline{\mathcal{RR}^0}}^T`.

Through principle of virtual work, one can show (B 16.50):

.. math::

   \underline{N}' = -\underline{f}

   \underline{M}'+ \left(\tilde{x}^{0\prime}+\tilde{u}'\right) \underline{N} = -\underline{m}

Sectional linear and angular momenta in material/inertial system (B 16.60):

.. math::

   \underline{\mathcal{P}}^* =
   \begin{Bmatrix}
   \underline{h}^* \\ \underline{g}^*
   \end{Bmatrix}
   =\underline{\underline{\mathcal{M}}}^* \underline{\mathcal{V}}^*

   \underline{\mathcal{P}} =
   \begin{Bmatrix}
   \underline{h} \\ \underline{g}
   \end{Bmatrix}
   =\underline{\underline{\mathcal{M}}}\, \underline{\mathcal{V}}

for :math:`\underline{\underline{\mathcal{M}}} \in \mathbb{R}^{6\times6}`,

Through Hamilton's principle, one can show (B 16.63):

.. math::

   \dot{\underline{h}} - \underline{N}' = \underline{f}

   \dot{g} + \dot{\tilde{u}} \underline{h} - \underline{M}'- \left(\tilde{x}'^0+\tilde{u}'\right) \underline{N} = \underline{m}

Finite Element Formulation
--------------------------

The inertial forces are (B 17.106):

.. math::

   \underline{\mathcal{F}}^I = \dot{\underline{\mathcal{P}}} +
   \begin{bmatrix} \underline{\underline{0}} & \underline{\underline{0}} \\ \underline{\underline{0}} & \dot{\tilde{u}}  \end{bmatrix}
   \underline{\mathcal{P}}

   \underline{\mathcal{F}}^I \in \mathbb{R}^6

which can be written (B 17.110) (note the term :math:`m \dot{\tilde{u}}\dot{\underline{u}}`, which arises in chain rule, is identically zero):

.. math::

   \underline{\mathcal{F}}^I = \begin{bmatrix}
   m \ddot{\underline{u}} +
   \left( \dot{\tilde{\omega}}+ \tilde{\omega} \tilde{\omega} \right) m \underline{\eta}\\
   m \tilde{\eta} \ddot{\underline{u}} + \underline{\underline{\rho}} \dot{\underline{\omega}}
    + \tilde{\omega} \underline{\underline{\rho}} \underline{\omega}
   \end{bmatrix}

where :math:`m`, :math:`\underline{\eta}`, and :math:`\underline{\underline{\rho}}` are readily extracted from the section mass matrix in inertial coordinates:

.. math::

   \underline{\underline{\mathcal{M}}} = \underline{\underline{\mathcal{RR}^0}}\, \underline{\underline{\mathcal{M}}}^*\, \underline{\underline{\mathcal{RR}^0}}^T =
   \begin{bmatrix}
   m \underline{\underline{I}}_3 & m \tilde{\eta}^T\\
   m \tilde{\eta} & \underline{\underline{\rho}}
   \end{bmatrix}

The elastic forces are (B 17.118):

.. math::

   \underline{\mathcal{F}}^C =
   \begin{Bmatrix} \underline{\mathcal{F}}^C_1 \\ \underline{\mathcal{F}}^C_2  \end{Bmatrix}=
   \begin{Bmatrix} \underline{N} \\ \underline{M} \end{Bmatrix} = \underline{\underline{\mathcal{C}}}\, \underline{e}

   \underline{\mathcal{F}}^D =
   \begin{Bmatrix} \underline{0} \\ \left(\tilde{x}'^0+\tilde{u}'\right)^T\underline{\mathcal{F}}^C_1 \end{Bmatrix} =
   \begin{Bmatrix} \underline{0} \\ \left(\tilde{x}'^0+\tilde{u}'\right)^T\underline{N} \end{Bmatrix}

   \underline{\mathcal{F}}^C, \underline{\mathcal{F}}^D \in \mathbb{R}^6

.. math::

   \underline{\underline{\mathcal{C}}} = \begin{bmatrix}
   \underline{\underline{\mathcal{C}}}_{11} & \underline{\underline{\mathcal{C}}}_{12} \\
   \underline{\underline{\mathcal{C}}}_{21} & \underline{\underline{\mathcal{C}}}_{22} \end{bmatrix}

.. math::

   \underline{e} = \begin{Bmatrix} \underline{e}_1 \\ \underline{e}_2  \end{Bmatrix}

   \underline{e}_1 = \underline{x}^{0\prime}+\underline{u}'-\left(\underline{\underline{R}}\,\underline{\underline{R}}^0\right) \overline{i}_1

   \underline{e}_2 = \underline{\kappa} = \mathrm{axial}\left( \underline{\underline{R}}' \underline{\underline{R}}^T \right)
   = 2 F(\hat{q}) \hat{q}^\prime

In the above for :math:`\underline{e}_1` there is a potential problem at non-node quadrature points. The tangent, :math:`\underline{x}^{0 \prime}`, is unlikely to match :math:`\underline{\underline{R}}^0 \overline{i}_1`; that mismatch would cause incorrect strain. We will use the following equivalent definition for :math:`\underline{e}_1`:

.. math::

   \underline{e}_1 = \underline{x}^{0 \prime} +\underline{u}'-\underline{\underline{R}}\, \underline{x}^{0 \prime}

which will guarantee no strain in reference configuration (i.e., :math:`\underline{e}_1 = 0` when :math:`\underline{u}^\prime=0` and :math:`\underline{\underline{R}}=\underline{\underline{I}}`).

Linearization
-------------

Inertial terms (see B 17.111)

.. math::

   \Delta \underline{\mathcal{F}}^I =
   \underline{\underline{\mathcal{M}}} \Delta \ddot{\underline{q}}
   + \underline{\underline{\mathcal{G}}} \Delta \dot{\underline{q}}
   + \underline{\underline{\mathcal{K}}} \Delta \underline{q}

where

.. math::

   \underline{\underline{\mathcal{G}}} =
   \begin{bmatrix}
   \underline{\underline{0}} & \widetilde{ \widetilde{\omega} m \underline{\eta} }^T
            + \widetilde{\omega} m \widetilde{\eta}^T\\
   \underline{\underline{0}} & \widetilde{\omega} \underline{\underline{\rho}} - \widetilde{\underline{\underline{\rho}} \underline{\omega}}
   \end{bmatrix}

.. math::

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

.. math::

   \Delta \underline{\mathcal{F}}^C =
   \underline{\underline{\mathcal{C}}} \Delta \underline{q}' + \underline{\underline{\mathcal{O}}} \Delta \underline{q}

.. math::

   \underline{\underline{\mathcal{O}}} =
   \begin{bmatrix}
   \underline{\underline{0}} &  -\widetilde{N} + \underline{\underline{\mathcal{C}}}_{11}\left(  \tilde{x}^{0 \prime}+ \tilde{u}' \right)   \\
   \underline{\underline{0}} &  -\widetilde{M} + \underline{\underline{\mathcal{C}}}_{21}\left( \tilde{x}^{0 \prime} + \tilde{u}' \right)
   \end{bmatrix}

.. math::

   \Delta \underline{q} = \begin{Bmatrix} \Delta \underline{u} \\ \underline{\Delta \psi} \end{Bmatrix}

   \Delta \underline{q}' = \begin{Bmatrix} \Delta \underline{u}' \\ \underline{\Delta \psi}' \end{Bmatrix}

.. math::

   \Delta \underline{\mathcal{F}}^D =
   \underline{\underline{\mathcal{P}}} \Delta \underline{q}' + \underline{\underline{\mathcal{Q}}} \Delta \underline{q}

.. math::

   \underline{\underline{\mathcal{P}}} =
   \begin{bmatrix}
   \underline{\underline{0}} & \underline{\underline{0}} \\
    \widetilde{N} + \left(  \tilde{x}^{0 \prime}+ \tilde{u}' \right)^T
   \underline{\underline{\mathcal{C}}}_{11} &
   \left( \tilde{x}^{0 \prime} + \tilde{u}' \right)^T
   \underline{\underline{\mathcal{C}}}_{12}
   \end{bmatrix}

.. math::

   \underline{\underline{\mathcal{Q}}} =
   \begin{bmatrix}
   \underline{\underline{0}} & \underline{\underline{0}} \\
    \underline{\underline{0}} &
   \left( \tilde{x}^{0 \prime} + \tilde{u}' \right)^T
   \left[-\widetilde{N} + \underline{C}_{11} \left( \tilde{x}^{0 \prime} + \tilde{u}' \right) \right]
   \end{bmatrix}

Finite-element representation:

.. math::

   \underline{q}(s) = \sum_{j=0}^{N} \phi_j(s) \widehat{\underline{q}}_j, \qquad \underline{q}, \widehat{\underline{q}}_j \in \mathbb{R}^7

Strong Form:

.. math::

   {\underline{\mathcal{F}}^{I}}
   -{\underline{\mathcal{F}}^{C}}'+\underline{\mathcal{F}}^D-\underline{\mathcal{F}}^G
   -\underline{\mathcal{F}}^{\mathrm{ext}} = \underline{0}

Weak Form (Residual):

.. math::

   \int_{-1}^{1} \phi_i \left(
   {\underline{\mathcal{F}}^{I}}
   -{\underline{\mathcal{F}}^{C}}'
   +\underline{\mathcal{F}}^D-\underline{\mathcal{F}}^G
   -\underline{\mathcal{F}}^{\mathrm{ext}}\right) J(s) dx = \underline{0}

.. math::

   \int_{-1}^{1} \left(
   J(s) \phi_i \underline{\mathcal{F}}^I
   + \phi'_i{\underline{\mathcal{F}}^{C}}+ J(s) \phi_i \underline{\mathcal{F}}^D\right)ds
   -\int_{-1}^{1} \phi_i\left(\underline{\mathcal{F}}^G
   +\underline{\mathcal{F}}^{\mathrm{ext}}\right) J(s) dx = \underline{0}, \quad \forall i\in\{0,1,\ldots,N\}

Elastic nodal force vector:

.. math::

   \underline{F}_i^E =
   \sum_{\ell=0}^{n^Q}
   \left[ \phi'_i(s_\ell)
   {\underline{\mathcal{F}}^{C}}(s_\ell)+ J(s_\ell) \phi_i(s_\ell) \underline{\mathcal{F}}^D(s_\ell) \right] w_\ell\, \quad
   \forall i\in\{0,1,\ldots,N\}\,, \\
   \underline{F}_i^E \in \mathbb{R}^6

Inertial nodal force vector:

.. math::

   \underline{F}_i^{I} =
   \sum_{\ell=0}^{n^Q}
   J(s_\ell) \phi_i(s_\ell) \underline{\mathcal{F}}^I(s_\ell) w_\ell\, \quad
   \forall i\in\{0,1,\ldots,N\}\,, \\
   \underline{F}_i^I \in \mathbb{R}^6

External nodal force vector:

.. math::

   \underline{F}_i^{ext} =
   \sum_{\ell=0}^{n^Q} \phi_i (s_\ell)
   \underline{F}^{ext}(s_\ell) J(s_\ell) w_\ell \,,
   \quad \forall i\in\{0,1,\ldots,N\}\,,\\
   \underline{F}_i^{ext} \in \mathbb{R}^6

Gravity nodal force vector:

.. math::

   \underline{F}_i^g =
   \sum_{\ell=0}^{n^Q} \phi_i (s_\ell) \underline{\mathcal{F}}^G(s_\ell)
   J(s_\ell) w_\ell, \,
   \quad \forall i\in\{0,1,\ldots,N\}\,,\\
   \underline{F}_i^g \in \mathbb{R}^6

Spectral finite-element portion of the residual vector:

.. math::

   \underline{R}^{FE} =
   \begin{bmatrix}
   \underline{{F}}_{0}^I + \underline{{F}}_{0}^{E} -
   \underline{{F}}_{0}^{ext} - \underline{{F}}_{0}^{g}\\
   \underline{{F}}_{1}^I + \underline{{F}}_{1}^{E} -
   \underline{{F}}_{1}^{ext} - \underline{{F}}_{1}^{g}\\
   \vdots \\
   \underline{{F}}_{N}^I + \underline{{F}}_{N}^{E} -
   \underline{{F}}_{N}^{ext} - \underline{{F}}_{N}^{g}\\
   \end{bmatrix}\,, \quad
   \underline{R}^{FE} \in \mathbb{R}^{6(N+1)}

Linearized form for finite-element (FE) portion of iteration matrix (inertial contributions):

.. math::

   \underline{\underline{M}}_{ij} =
   \sum_{\ell=0}^{n^Q} &
   \phi_i(s_\ell) \underline{\underline{\mathcal{M}}}(s_\ell) \phi_j(s_\ell) J(s_\ell) w_\ell \\
   &\forall i,j\in\{0,1,\ldots,N\}
   , \quad \underline{\underline{M}}_{ij} \in \mathbb{R}^{6\times 6}

.. math::

   \underline{\underline{G}}_{ij} =
   \sum_{\ell=0}^{n^Q} &
   \phi_i(s_\ell) \underline{\underline{\mathcal{G}}}(s_\ell) \phi_j(s_\ell) J(s_\ell) w_\ell \\
   &\forall i,j\in\{0,1,\ldots,N\}
   , \quad \underline{\underline{G}}_{ij} \in \mathbb{R}^{6\times 6}

.. math::

   \underline{\underline{K}}_{ij}^I =
   \sum_{\ell=0}^{n^Q} &
   \phi_i(s_\ell) \underline{\underline{\mathcal{K}}}(s_\ell) \phi_j(s_\ell) J(s_\ell) w_\ell \\
   &\forall i,j\in\{0,1,\ldots,N\}
   , \quad \underline{\underline{K}}_{ij}^I \in \mathbb{R}^{6\times 6}

Linearized form for finite-element (FE) portion of iteration matrix (elastic contribution):

.. math::

   \underline{\underline{K}}_{ij}^E =
   \sum_{\ell=0}^{n^Q} \big[&
   \phi_i(s_\ell) \underline{\underline{\mathcal{P}}}(s_\ell) \phi'_j(s_\ell) +
   \phi_i(s_\ell) \underline{\underline{\mathcal{Q}}}(s_\ell) \phi_j(s_\ell) J(s_\ell)+ \\
   &
   \phi'_i(s_\ell) \underline{\underline{\mathcal{C}}}(s_\ell) \phi'_j(s_\ell) \frac{1}{J(s_\ell)}+
   \phi'_i(s_\ell) \underline{\underline{\mathcal{O}}}(s_\ell) \phi_j(s_\ell)
   \big] w_\ell \\
   &\forall i,j\in\{0,1,\ldots,N\}
   , \quad \underline{\underline{K}}_{ij}^E \in \mathbb{R}^{6\times 6}

Spectral finite-element portion of the iteration matrix:

.. math::

   \underline{\underline{K}}^{FE} =
   \begin{bmatrix}
   \underline{\underline{K}}_{00}^I + \underline{\underline{K}}_{00}^E&
   \underline{\underline{K}}_{01}^I + \underline{\underline{K}}_{01}^E& \ldots &
   \underline{\underline{K}}_{0N}^I + \underline{\underline{K}}_{0N}^E\\
   \underline{\underline{K}}_{10}^I + \underline{\underline{K}}_{10}^E &
    \underline{\underline{K}}_{11}^I  + \underline{\underline{K}}_{11}^E&
   \ldots &
   \underline{\underline{K}}_{1N}^I + \underline{\underline{K}}_{1N}^E\\
   \vdots & \vdots & \vdots & \vdots \\
   \underline{\underline{K}}_{N0}^I + \underline{\underline{K}}_{N0}^E&
   \underline{\underline{K}}_{N1}^I + \underline{\underline{K}}_{N1}^E& \ldots &
   \underline{\underline{K}}_{NN}^I + \underline{\underline{K}}_{NN}^E\\
   \end{bmatrix}, \,
   \underline{\underline{K}}^{FE} \in \mathbb{R}^{6 (N+1) \times 6 (N+1)}

.. math::

   \underline{\underline{M}}^{FE} =
   \begin{bmatrix}
   \underline{\underline{M}}_{00}&
   \underline{\underline{M}}_{01}& \ldots &
   \underline{\underline{M}}_{0N}\\
   \underline{\underline{M}}_{10} &
   \underline{\underline{M}}_{11}&
   \ldots &
   \underline{\underline{M}}_{1N}\\
   \vdots & \vdots & \vdots & \vdots \\
   \underline{\underline{M}}_{N0}&
   \underline{\underline{M}}_{N1}& \ldots &
   \underline{\underline{M}}_{NN}\\
   \end{bmatrix},\,
   \underline{\underline{M}}^{FE} \in \mathbb{R}^{6 (N+1) \times 6 (N+1)}

.. math::

   \underline{\underline{G}}^{FE} =
   \begin{bmatrix}
   \underline{\underline{G}}_{00}&
   \underline{\underline{G}}_{01}& \ldots &
   \underline{\underline{G}}_{0N}\\
   \underline{\underline{G}}_{10} &
   \underline{\underline{G}}_{11}&
   \ldots &
   \underline{\underline{G}}_{1N}\\
   \vdots & \vdots & \vdots & \vdots \\
   \underline{\underline{G}}_{N0}&
   \underline{\underline{G}}_{N1}& \ldots &
   \underline{\underline{G}}_{NN}\\
   \end{bmatrix},\,
   \underline{\underline{G}}^{FE} \in \mathbb{R}^{6 (N+1) \times 6 (N+1)}

Constraints (zero subscripts denote the zeroth/root nodal value):

.. math::

   \underline{\Phi} =
   \begin{Bmatrix}
   \underline{u}_0- \underline{u}_\mathrm{BC} \\ \underline{\psi}_0-\underline{\psi}_\mathrm{BC}
   \end{Bmatrix}, \quad \underline{\Phi} \in \mathbb{R}^6

.. math::

   \underline{F}^C = \underline{\underline{B}}^T \underline{\lambda}
   \,, \quad \underline{F}^C \in \mathbb{R}^{6 (N+1)}, \quad \underline{\lambda} \in \mathbb{R}^6

.. math::

   \underline{\underline{B}} =
   \begin{bmatrix}
   \underline{\underline{I}} & \underline{\underline{0}} &  \underline{\underline{0}} &\ldots & \underline{\underline{0}} \\
   \underline{\underline{0}} & \underline{\underline{I}} & \underline{\underline{0}} &\ldots & \underline{\underline{0}}
   \end{bmatrix}\,, \quad \underline{\underline{B}} \in \mathbb{R}^{6\times 6 (N+1)}

.. math::

   \underline{\underline{K}}^C =
   \begin{bmatrix}
   \underline{\underline{0}} & \underline{\underline{0}} & \ldots & \underline{\underline{0}} \\
   \vdots & \vdots & \ldots & \vdots \\
   \underline{\underline{0}} & \underline{\underline{0}} & \ldots & \underline{\underline{0}}
   \end{bmatrix} \,, \quad \underline{\underline{K}}^C
   \in \mathbb{R}^{\{6 (N+1)\}\times \{6 (N+1)\}}

Full residual:

.. math::

   \underline{\underline{R}} =
   \begin{Bmatrix}
   \underline{R}^{FE} + \underline{F}^C \\
   \underline{\Phi}
   \end{Bmatrix}
   , \quad \underline{R}_{t} \in \mathbb{R}^{6(N+1)+6}

Iteration matrix:

.. math::

   \underline{\underline{S}}_t =
   \begin{bmatrix}
   \underline{\underline{M}}^{FE} \beta'+\underline{\underline{G}}^{FE} \gamma' + \left(\underline{\underline{K}}^{FE} + \underline{\underline{K}}^C\right)
   \underline{\underline{T}}(h \Delta q) & \underline{\underline{B}}^T \\
   \underline{\underline{B}}\,\underline{\underline{T}}(h \Delta q)                     & \underline{\underline{0}}
   \end{bmatrix}
   , \quad \underline{\underline{S}}_{t} \in \mathbb{R}^{\{6(N+1)+6\} \times \{6(N+1)+6\}}

Implementation
--------------

To maximize performance in assembling the iteration matrix, the tangent matrix is applied to the stiffness matrix and constraints matrix by exploiting the block-diagonal structure of the tangent matrix.  In particular, the tangent matrix can be represented as a block-diagonal matrix where each block is a :math: '6x6' matrix corresponding to the degrees of freedom at each node.  Similarly, the system matrix can be interpreted as a block matrix of :math: '(N+1)x(N+1)' blocks of each size :math: '6x6'. To multiply these matrices together efficiently, you can multiply the jth block-column in the ith block-row by the jth block-diagonal entry of the tangent matrix, with each block-multiplication being a 6x6 matrix multiplication.

.. math::

   \left(\left(\underline{\underline{K}}^{FE} + \underline{\underline{K}}^C\right) \underline{\underline{T}}(h \Delta q)\right)_{i,j} =
   \left(underline{\underline{K}}^{Block}\right)_{i,j} \left(\underline{\underline{T}}^{Block}\right)_{j, j}

The performance of this simplification is further improved by only applying the multiplication for the blocks which are non-zero in the stiffness matrix.  Specifically, these are the block-columns corresponding to the nodes which share a beam element with the node corresponding to the current block row.  A similar technique is utilized when applying the tangent matrix to the constraints matrix
