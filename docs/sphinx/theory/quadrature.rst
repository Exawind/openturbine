.. _sec-quadrature:

Notes on quadrature
^^^^^^^^^^^^^^^^^^^

In Kynema, we represent beams that can have highly variable
properties along the length and we use a single high-order element for
the whole beam. Material properties, including the sectional mass matrix
:math:`\underline{\underline{M}}^*`, stiffness matrix
:math:`\underline{\underline{C}}^*`, and twist :math:`\tau`, are defined
by the user at stations located along the beam reference line. In
typical finite-element beam implementations, :math:`P`-point
Gauss-Legendre quadrature is common. While that is often sufficient for
uniform or linearly varying properties, it can be inadequate for highly
variable material properties. Kynema provides users the option of
Guass-Legendre or trapezoid-rule quadrature, with the former being
suitable for constant-material-property beams and the latter for
variable-property beams. Trapezoid-quadrature locations at each
user-defined material point, and additional points are added between
points in a user-defined manner that divides sections between
user-defined propery points in an equidistant fashion.
