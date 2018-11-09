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

from numpy import isclose, logical_and
from dolfin import *
from dolfin.fem import assemble
from multiphenics import *

"""
In this tutorial we first solve the problem

-u'' = f    in Omega = [0, 1]
 u   = 0    on Gamma = {0, 1}
 
using standard FEniCS code.

Then we use multiphenics to solve the system

-   w_1'' - 2 w_2'' = 3 f    in Omega
- 3 w_1'' - 4 w_2'' = 7 f    in Omega

subject to

 w_1 = 0    on Gamma = {0, 1}
 w_2 = 0    on Gamma = {0, 1}
 
By construction the solution of the system is
    (w_1, w_2) = (u, u)

We then compare the solution provided by multiphenics
to the one provided by standard FEniCS.
"""

# Mesh generation
mesh = UnitIntervalMesh(MPI.comm_world, 32)

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return logical_and(abs(x[:, 0] - 0.) < DOLFIN_EPS, on_boundary)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return logical_and(abs(x[:, 0] - 1.) < DOLFIN_EPS, on_boundary)
        
boundaries = MeshFunction("size_t", mesh, mesh.topology.dim - 1, 0)
left = Left()
left.mark(boundaries, 1)
right = Right()
right.mark(boundaries, 1)

x0 = SpatialCoordinate(mesh)[0]

# Solver parameters
solver_parameters = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}

def run_standard():
    # Define a function space
    V = FunctionSpace(mesh, ("Lagrange", 2))
    u = TrialFunction(V)
    v = TestFunction(V)

    # Define problems forms
    a = inner(grad(u), grad(v))*dx + u*v*dx
    f = 100*sin(20*x0)*v*dx

    # Define boundary conditions
    zero = Function(V)
    zero.vector().set(0.0)
    bc = DirichletBC(V, zero, (boundaries, 1))
    
    # Solve the linear system
    u = Function(V)
    solve(a == f, u, bc, petsc_options=solver_parameters)
    
    # Return the solution
    return u
    
u = run_standard()

def run_block():
    # Define a block function space
    V = FunctionSpace(mesh, ("Lagrange", 2))
    VV = BlockFunctionSpace([V, V])
    uu = BlockTrialFunction(VV)
    vv = BlockTestFunction(VV)
    (u1, u2) = block_split(uu)
    (v1, v2) = block_split(vv)

    # Define problem block forms
    aa = [[1*inner(grad(u1), grad(v1))*dx + 1*u1*v1*dx, 2*inner(grad(u2), grad(v1))*dx + 2*u2*v1*dx],
          [3*inner(grad(u1), grad(v2))*dx + 3*u1*v2*dx, 4*inner(grad(u2), grad(v2))*dx + 4*u2*v2*dx]]
    ff = [300*sin(20*x0)*v1*dx,
          700*sin(20*x0)*v2*dx]
    
    # Define block boundary conditions
    zero = Function(V)
    zero.vector().set(0.0)
    bc1 = DirichletBC(VV.sub(0), zero, (boundaries, 1))
    bc2 = DirichletBC(VV.sub(1), zero, (boundaries, 1))
    bcs = BlockDirichletBC([bc1,
                            bc2])
    
    # Solve the block linear system
    uu = BlockFunction(VV)
    block_solve(aa, uu.block_vector(), ff, bcs, petsc_options=solver_parameters)
    uu1, uu2 = uu
    
    # Return the block solution
    return uu1, uu2
    
uu1, uu2 = run_block()

u_norm = sqrt(assemble(inner(grad(u), grad(u))*dx))
err_1_norm = sqrt(assemble(inner(grad(u - uu1), grad(u - uu1))*dx))
err_2_norm = sqrt(assemble(inner(grad(u - uu2), grad(u - uu2))*dx))
print("Relative error for first component is equal to", err_1_norm/u_norm)
print("Relative error for second component is equal to", err_2_norm/u_norm)
assert isclose(err_1_norm/u_norm, 0., atol=1.e-10)
assert isclose(err_2_norm/u_norm, 0., atol=1.e-10)
