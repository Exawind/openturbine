R3-based geometrically nonlinear, constitutively linear spring
==============================================================

**Overview:** Geometrically nonlinear and constitutively linear spring defined in :math:`\underline{\underline{R}}^3`

**Spring Model Input at Initialization:**

- :math:`\underline{x}_1^0 \in \mathbb{R}^3`: node 1 reference position; inertial CS
- :math:`\underline{x}_2^0 \in \mathbb{R}^3`: node 2 reference position; inertial CS
- :math:`\underline{u}_1^i \in \mathbb{R}^3`: node 1 initial displacement
- :math:`\underline{u}_2^i \in \mathbb{R}^3`: node 2 initial displacement
- :math:`k` constitutively linear spring stiffness
- :math:`\ell^\mathrm{us}` unstretched reference length

**Degrees of Freedom:**

- :math:`\underline{u}_1 \in \mathbb{R}^3`: node 1 displacement at time :math:`t`
- :math:`\underline{u}_2 \in \mathbb{R}^3`: node 2 displacement at time :math:`t`

**Force:**

.. math::

   \underline{F} = \begin{bmatrix}
   \underline{f} \\
   -\underline{f}
   \end{bmatrix}
   \,, \mathrm{where}\, \underline{F} \in \mathbb{R}^6

and

.. math::

   \underline{f} = -k \frac{\underline{r} }{| \underline{r} |} \left( | \underline{r} | - \ell^\mathrm{us} \right)
   \,, \mathrm{where}\, \underline{r} = \underline{x}_2^0 + \underline{u}_2 - \underline{x}_1^0  - \underline{u}_1

.. math::

   \delta \underline{f} = \underline{\underline{K}}_t
   \begin{bmatrix}
   \delta \underline{u}_1\\
   \delta \underline{u}_2
   \end{bmatrix}

.. math::

   \underline{\underline{K}}_t =  \begin{bmatrix}
   \underline{\underline{A}} & -\underline{\underline{A}} \\
   - \underline{\underline{A}} & \underline{\underline{A}}
   \end{bmatrix}

and

.. math::

   \underline{\underline{A}} =  k \left( \frac{\ell^\mathrm{us} }{|\underline{r} |} - 1\right) \underline{\underline{I}}
   - \frac{k \ell^\mathrm{us}}{|\underline{r}|^3}\left( \widetilde{r} \widetilde{r} + \underline{\underline{I}} (\underline{r}^T \underline{r} ) \right)

Aside: For :math:`\underline{a},\underline{r}\in \mathbb{R}^3`,
:math:`\underline{r} (\underline{a}^T \underline{r}) =
\left[ \widetilde{r} \widetilde{r} + \underline{\underline{I}} \left( \underline{r}^T \underline{r} \right)
\right] \underline{a}`
