R3xSO(3)-based Rigid Body
==========================

Overview
--------

Details for a rigid-body solver. Unlike Bruls et al (2012), this is formulated in inertial coordinates.

Rigid-Body Model Input at Initialization
----------------------------------------

- **M**\* ∈ ℝ\ :sup:`6×6`: Rigid body mass matrix in material coordinate system (CS)
- **x**\ :sup:`0` ∈ ℝ\ :sup:`3`: Rigid body reference position; inertial CS
- **R**\ :sup:`0` ∈ ℝ\ :sup:`3×3`: Rigid body reference orientation
- **u**\ :sup:`i` ∈ ℝ\ :sup:`3`: Rigid body initial displacement
- **R**\ :sup:`i` ∈ ℝ\ :sup:`3×3`: Rigid body initial orientation
- **u̇**\ :sup:`i` ∈ ℝ\ :sup:`3`: Rigid body initial velocity
- **ω**\ :sup:`i` ∈ ℝ\ :sup:`3`: Rigid body initial angular velocity

Degrees of Freedom
------------------

- **u** ∈ ℝ\ :sup:`3`: Rigid body displacement at time *t*
- **R** ∈ ℝ\ :sup:`3×3`: Rigid-body orientation at time *t* (represented as **q̂** ∈ ℝ\ :sup:`4` quaternion)
- **ω** ∈ ℝ\ :sup:`3`: Rigid body angular velocity at time *t*
- Generalized coordinates:

.. math::

   q = \begin{bmatrix}
   u \\
   R \\
   \end{bmatrix}
   , \quad
   \dot{q} = \begin{bmatrix}
   \dot{u} \\
   \omega \\
   \end{bmatrix}
   , \quad
   \ddot{q} = \begin{bmatrix}
   \ddot{u} \\
   \dot{\omega} \\
   \end{bmatrix}

Residual
--------

.. math::

   R^{\mathrm{rb}} =
   \begin{bmatrix}
   M^{\mathrm{rb}}\, \ddot{q} + g^{\mathrm{rb}} - F^{\mathrm{rb}} + B^T\lambda \\
   \Phi
   \end{bmatrix}
   \in \mathbb{R}^{6+nc}

where

.. math::

   B \in \mathbb{R}^{nc \times 6}\,, \quad
   \lambda \in \mathbb{R}^{nc}\,, \quad
   \Phi \in \mathbb{R}^{nc}\,, \quad
   F^{\mathrm{rb}}
   =
   \begin{bmatrix}
   f\\
   m\\
   \end{bmatrix}
   \in \mathbb{R}^{6}

.. math::

   M^{\mathrm{rb}}
   = \begin{bmatrix}
   m I_3 & m \tilde{\eta}^T \\
   m \tilde{\eta} & \rho
   \end{bmatrix}
   = \mathcal{RR}^0\, M^* {\mathcal{RR}^0}^T
   %
   \in \mathbb{R}^{6\times 6}
   \,,\quad
   g^{\mathrm{rb}} = \begin{bmatrix}
   m \tilde{\omega} \tilde{\omega} \eta  \\
   \tilde{\omega} \rho \omega
   \end{bmatrix}

and where

.. math::

   \mathcal{RR}^0 =
   \begin{bmatrix}
   R^0 & 0 \\
   0 & R^0
   \end{bmatrix}
   \quad \text{for } \mathcal{RR}^0 \in \mathbb{R}^{6\times6}

Time Integration
----------------

Iteration matrix:

.. math::

   S_t =
   \begin{bmatrix}
   M_t \beta'+C_t \gamma' + K_t
   T(h \Delta q) & B^T \\
   B\,T(h \Delta q)                     & 0
   \end{bmatrix}
   , \quad S_{t} \in \mathbb{R}^{(6+nc) \times (6+nc)}

where

.. math::

   M_t = M^{\mathrm{rb}}\in \mathbb{R}^{6\times 6} \,, \quad
   C_t = \mathcal{G}^{\mathrm{rb}}\in \mathbb{R}^{6\times 6} \,, \quad
   K_t = \mathcal{K}^{\mathrm{rb}} \in \mathbb{R}^{6\times 6}

and where

.. math::

   \mathcal{G}^{\mathrm{rb}} =
   \begin{bmatrix}
   0 & \widetilde{ \widetilde{\omega} m \eta }^T
            + \widetilde{\omega} m \widetilde{\eta}^T\\
   0 & \widetilde{\omega} \rho - \widetilde{\rho \omega}
   \end{bmatrix}

.. math::

   \mathcal{K}^{\mathrm{rb}} =
   \begin{bmatrix}
   0 & \left( \dot{\widetilde{\omega}} + \tilde{\omega}\tilde{\omega}
           \right) m \widetilde{\eta}^T\\
   0 & \ddot{\widetilde{u}} m \widetilde{\eta}
            + \left(\rho\dot{\widetilde{\omega}}
                    -\widetilde{\rho \dot{\omega}} \right)
            + \widetilde{\omega} \left( \rho \widetilde{\omega}
            - \widetilde{ \rho\omega} \right)
   \end{bmatrix}
