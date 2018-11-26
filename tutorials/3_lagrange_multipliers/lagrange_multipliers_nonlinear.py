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

import numpy
from numpy import isclose
from ufl import replace
from dolfin import *
from dolfin.cpp.mesh import GhostMode
from multiphenics import *
from multiphenics.io import XDMFFile

r"""
In this example we solve a nonlinear Laplace problem associated to
    min E(u)
    s.t. u = g on \partial \Omega
where
    E(u) = \int_\Omega { (1 + u^2) |grad u|^2 - u } dx
using a Lagrange multiplier to handle non-homogeneous Dirichlet boundary conditions.
"""

# MESHES #
# Mesh
mesh = XDMFFile(MPI.comm_world, "data/circle.xdmf").read_mesh(MPI.comm_world, GhostMode.none)
subdomains = XDMFFile(MPI.comm_world, "data/circle_physical_region.xdmf").read_mf_size_t(mesh)
boundaries = XDMFFile(MPI.comm_world, "data/circle_facet_region.xdmf").read_mf_size_t(mesh)
# Dirichlet boundary
boundary_restriction = XDMFFile(MPI.comm_world, "data/circle_restriction_boundary.rtc.xdmf").read_mesh_restriction(mesh)

# FUNCTION SPACES #
# Function space
V = FunctionSpace(mesh, "Lagrange", 2)
# Block function space
W = BlockFunctionSpace([V, V], restrict=[None, boundary_restriction])

# TRIAL/TEST FUNCTIONS #
dul = BlockTrialFunction(W)
(du, dl) = block_split(dul)
ul = BlockFunction(W)
(u, l) = block_split(ul)
vm = BlockTestFunction(W)
(v, m) = block_split(vm)

# MEASURES #
dx = Measure("dx")(subdomain_data=subdomains)
ds = Measure("ds")(subdomain_data=boundaries)

# ASSEMBLE #
x = SpatialCoordinate(mesh)
g = sin(3*x[0] + 1)*sin(3*x[1] + 1)
F = [inner((1+u**2)*grad(u), grad(v))*dx + u*v*inner(grad(u), grad(u))*dx + l*v*ds - v*dx,
     u*m*ds - g*m*ds]
J = block_derivative(F, ul, dul)

# SOLVE #
problem = BlockNonlinearProblem(F, ul, None, J)
solver = BlockNewtonSolver(problem)
solver_parameters = {"maximum_iterations": 20, "report": True}
solver.parameters.update(solver_parameters)
solver.solve(problem, ul.block_vector())

# ERROR #
u_ex = Function(V)
F_ex = replace(F[0], {u: u_ex})
J_ex = derivative(F_ex, u_ex, du)
@function.expression.numba_eval
def g_eval(values, x, cell):
    values[:, 0] = numpy.sin(3*x[:, 0] + 1)*numpy.sin(3*x[:, 1] + 1)
bc_ex = DirichletBC(V, interpolate(Expression(g_eval), V), (boundaries, 1))
problem_ex = NonlinearVariationalProblem(F_ex, u_ex, bc_ex, J_ex)
solver_ex = NonlinearVariationalSolver(problem_ex)
solver_ex.parameters.update(snes_solver_parameters)
solver_ex.solve()
err = Function(V)
err.vector().add_local(+ u_ex.vector().get_local())
err.vector().add_local(- u.vector().get_local())
err.vector().apply("")
u_ex_norm = sqrt(assemble(inner(grad(u_ex), grad(u_ex))*dx))
err_norm = sqrt(assemble(inner(grad(err), grad(err))*dx))
print("Relative error is equal to", err_norm/u_ex_norm)
assert isclose(err_norm/u_ex_norm, 0., atol=1.e-10)
