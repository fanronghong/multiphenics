# Copyright (C) 2016-2019 by the multiphenics authors
#
# This file is part of multiphenics.
#
# multiphenics is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multiphenics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

from numpy import isclose
from dolfin import *
from dolfin import function
import matplotlib.pyplot as plt
from multiphenics import *

r"""
In this tutorial we solve the optimal control problem

min J(y, u) = 1/2 \int_{\Omega} (y - y_d)^2 dx + \alpha/2 \int_{\Omega} u^2 dx
s.t.
    - \Delta y = f + u   in \Omega
             y = 1       on \partial \Omega
             
where
    \Omega                      unit square
    u \in L^2(\Omega)           control variable
    y \in H^1_0(\Omega)         state variable
    \alpha > 0                  penalization parameter
    y_d = piecewise constant    desired state
    f                           forcing term
    
using an adjoint formulation solved by a one shot approach.
The test case is from section 5.1 of
F. Negri, G. Rozza, A. Manzoni and A. Quarteroni. Reduced Basis Method for Parametrized Elliptic Optimal Control Problems. SIAM Journal on Scientific Computing, 35(5): A2316-A2340, 2013.
"""

# MESH #
mesh = Mesh("data/rectangle.xml")
subdomains = MeshFunction("size_t", mesh, "data/rectangle_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/rectangle_facet_region.xml")

# FUNCTION SPACES #
Y = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
U = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
Q = Y
W_el = BlockElement(Y, U, Q)
W = BlockFunctionSpace(mesh, W_el)

# PROBLEM DATA #
alpha = 0.01
y_d_1 = 1.0
y_d_2 = 0.6
f = 0.
@function.expression.numba_eval
def zero_eval(values, x, cell):
    values[:] = 0.0
bc0 = interpolate(Expression(zero_eval), W.sub(0))
@function.expression.numba_eval
def one_eval(values, x, cell):
    values[:] = 0.0
bc1 = interpolate(Expression(one_eval), W.sub(0))

# TRIAL/TEST FUNCTIONS #
yup = BlockTrialFunction(W)
(y, u, p) = block_split(yup)
zvq = BlockTestFunction(W)
(z, v, q) = block_split(zvq)

# MEASURES #
dx = Measure("dx")(subdomain_data=subdomains)

# OPTIMALITY CONDITIONS #
a = [[y*z*dx                    , 0           , inner(grad(p), grad(z))*dx],
     [0                         , alpha*u*v*dx, - p*v*dx                  ],
     [inner(grad(y), grad(q))*dx, - u*q*dx    , 0                         ]]
f =  [y_d_1*z*dx(1) + y_d_2*z*dx(2),
      0                            ,
      f*q*dx                        ]
bc = BlockDirichletBC([[DirichletBC(W.sub(0), bc1, boundaries, 1)],
                       [],
                       [DirichletBC(W.sub(2), bc0, boundaries, 1)]])

# SOLUTION #
yup = BlockFunction(W)
(y, u, p) = block_split(yup)

# FUNCTIONAL #
J = 0.5*inner(y - y_d_1, y - y_d_1)*dx(1) + 0.5*inner(y - y_d_2, y - y_d_2)*dx(2) + 0.5*alpha*inner(u, u)*dx

# UNCONTROLLED FUNCTIONAL VALUE #
A_state = assemble(a[2][0])
F_state = assemble(f[2])
[bc_state.apply(A_state) for bc_state in bc[0]]
[bc_state.apply(F_state) for bc_state in bc[0]]
solve(A_state, y.vector(), F_state)
print("Uncontrolled J =", assemble(J))
assert isclose(assemble(J), 0.24)
plt.figure()
plot(y, title="uncontrolled state")
plt.show()

# OPTIMAL CONTROL #
A = block_assemble(a, keep_diagonal=True)
F = block_assemble(f)
bc.apply(A)
bc.apply(F)
block_solve(A, yup.block_vector(), F)
print("Optimal J =", assemble(J))
assert isclose(assemble(J), 0.158474070)
plt.figure()
plot(y, title="state")
plt.figure()
plot(u, title="control")
plt.figure()
plot(p, title="adjoint")
plt.show()
