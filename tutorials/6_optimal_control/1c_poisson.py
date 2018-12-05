# Copyright (C) 2016-2018 by the multiphenics authors
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

import os
from numpy import isclose
from petsc4py import PETSc
from dolfin import *
import matplotlib.pyplot as plt
import multiphenics
from multiphenics import *

r"""
In this tutorial we solve the optimal control problem

min J(y, u) = 1/2 \int_{\Omega_1} (y - y_d)^2 dx + \alpha/2 \int_{\Omega} u^2 dx
s.t.
    - \Delta y = f + u   in \Omega
             y = 0       on \partial \Omega
             
where
    \Omega                      unit square
    \Omega_1 \subset \Omega     observation domain
    u \in L^2(\Omega)           control variable
    y \in H^1_0(\Omega)         state variable
    \alpha > 0                  penalization parameter
    y_d \in H^1(\Omega_1)       desired state
    f                           forcing term
    
using an adjoint formulation solved by a one shot approach. The resulting linear system is solved either with a direct linear solver or with a fieldsplict preconditioner.
The test case is from section 3 of
M. Stoll and A. Wathen, All-at-once solution of time-dependent PDE-constrained optimization problems, Oxford Centre for Collaborative Applied Mathematics, Technical Report 10/47, 2010.
"""

# MESH #
mesh = Mesh("data/cross.xml")
subdomains = MeshFunction("size_t", mesh, "data/cross_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/cross_facet_region.xml")

# FUNCTION SPACES #
Y = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
U = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
Q = Y
W_el = BlockElement(Y, U, Q)
W = BlockFunctionSpace(mesh, W_el)

# PROBLEM DATA #
alpha = Constant(0.01)
y_d = Expression("-x[0]*exp(- (x[0] - 0.5)*(x[0] - 0.5) - (x[1] - 0.5)*(x[1] - 0.5))", element=W.sub(0).ufl_element())
f = Constant(0.)

# TRIAL/TEST FUNCTIONS #
yup = BlockTrialFunction(W)
(y, u, p) = block_split(yup)
zvq = BlockTestFunction(W)
(z, v, q) = block_split(zvq)

# MEASURES #
dx = Measure("dx")(subdomain_data=subdomains)

# OPTIMALITY CONDITIONS #
a = [[inner(y, z)*dx(1),          0,                    inner(grad(p), grad(z))*dx],
     [0,                          alpha*inner(u, v)*dx, - inner(p, v)*dx          ],
     [inner(grad(y), grad(q))*dx, - inner(u, q)*dx,     0                         ]]
ap = [[inner(y, z)*dx,             0,                    inner(grad(p), grad(z))*dx],
      [0,                          alpha*inner(u, v)*dx, - inner(p, v)*dx          ],
      [inner(grad(y), grad(q))*dx, - inner(u, q)*dx,     0                         ]]
f =  [inner(y_d, z)*dx(1),
      0                  ,
      f*q*dx             ]
bc = BlockDirichletBC([[DirichletBC(W.sub(0), Constant(0.), boundaries, 1)],
                       [],
                       [DirichletBC(W.sub(2), Constant(0.), boundaries, 1)]])

# SOLUTION #
yup = BlockFunction(W)
(y, u, p) = block_split(yup)

# FUNCTIONAL #
J = 0.5*inner(y - y_d, y - y_d)*dx(1) + 0.5*alpha*inner(u, u)*dx

# UNCONTROLLED FUNCTIONAL VALUE #
A_state = assemble(a[2][0])
F_state = assemble(f[2])
[bc_state.apply(A_state) for bc_state in bc[0]]
[bc_state.apply(F_state) for bc_state in bc[0]]
solve(A_state, y.vector(), F_state)
print("Uncontrolled J =", assemble(J))
assert isclose(assemble(J), 0.04394608792218948)
plt.figure()
plot(y, title="uncontrolled state")
plt.show()

# OPTIMAL CONTROL #
# Assemble matrices
A = block_assemble(a, keep_diagonal=True)
AP = block_assemble(ap, keep_diagonal=True)
F = block_assemble(f)
bc.apply(A)
bc.apply(AP)
bc.apply(F)

# Preconditioner for the Schur complement
class SchurComplementPreconditioner(object):
    def __init__(self):
        # Restricted function spaces
        W_Y = W.extract_block_sub_space((0, ))
        W_U = W.extract_block_sub_space((1, ))
        W_Q = W.extract_block_sub_space((2, ))
        # System matrices
        k = block_restrict(ap, [W_Q, W_Y])
        k_bcs = block_restrict(bc, W_Y)
        K = block_assemble(k)
        k_bcs.apply(K)
        self.k_bcs = k_bcs
        k_t = block_restrict(ap, [W_Y, W_Q])
        k_t_bcs = block_restrict(bc, W_Q)
        K_T = block_assemble(k_t)
        k_t_bcs.apply(K_T)
        self.k_t_bcs = k_t_bcs
        m = block_restrict(ap, [W_Q, W_U])
        M = block_assemble(m)
        self.M = M
        # Solution vector storage
        self.work0 = BlockFunction(W_Q).block_vector()
        self.work1 = BlockFunction(W_Y).block_vector()
        self.work2 = BlockFunction(W_Y).block_vector()
        self.work3 = BlockFunction(W_Q).block_vector()
        # PETSc options for direct solvers
        direct_opt = PETSc.Options()
        direct_opt.setValue("direct_ksp_type", "preonly")
        direct_opt.setValue("direct_pc_type", "lu")
        direct_opt.setValue("direct_pc_factor_mat_solver_package", "mumps")
        # KSP objects
        ksp1 = PETSc.KSP().create(mesh.mpi_comm())
        ksp1.setOptionsPrefix("direct_")
        ksp1.setFromOptions()
        ksp1.setOperators(K.mat())
        ksp1.setUp()
        self.ksp1 = ksp1
        ksp3 = PETSc.KSP().create(mesh.mpi_comm())
        ksp3.setOptionsPrefix("direct_")
        ksp3.setFromOptions()
        ksp3.setOperators(K_T.mat())
        ksp3.setUp()
        self.ksp3 = ksp3
        
    def apply(self, pc, x, y):
        """
        Apply Sp^-1, where Sp = K M^-1 K^T.
        """
        # Wrap x into a multiphenics object
        x.copy(result=self.work0.vec())
        # store K^{-1}*x in work1
        self.k_bcs.apply(self.work0)
        self.ksp1.solve(self.work0.vec(), self.work1.vec())
        # store M*work1 in work2
        self.M.mat().mult(self.work1.vec(), self.work2.vec())
        # store K^{-T}*work2 in work3
        self.k_t_bcs.apply(self.work2)
        self.ksp3.solve(self.work2.vec(), self.work3.vec())
        # Copy work3 into y
        self.work3.vec().copy(result=y)
        
# Iterative solver class
class IterativeSolver(object):
    def __init__(self):
        # PETSc options for fieldsplit solver
        iter_opt = PETSc.Options()
        iter_opt.setValue("iter_ksp_monitor", "")
        iter_opt.setValue("iter_fieldsplit_yu_ksp_monitor", "")
        iter_opt.setValue("iter_fieldsplit_p_ksp_monitor", "")
        iter_opt.setValue("iter_ksp_type", "minres")
        iter_opt.setValue("iter_pc_type", "fieldsplit")
        iter_opt.setValue("iter_pc_fieldsplit_type", "schur")
        iter_opt.setValue("iter_pc_fieldsplit_schur_fact_type", "diag")
        iter_opt.setValue("iter_pc_fieldsplit_schur_precondition", "user")
        iter_opt.setValue("iter_fieldsplit_yu_ksp_type", "gmres")
        iter_opt.setValue("iter_fieldsplit_yu_pc_type", "lu")
        iter_opt.setValue("iter_fieldsplit_yu_pc_factor_mat_solver_package", "mumps")
        iter_opt.setValue("iter_fieldsplit_p_ksp_type", "gmres")
        iter_opt.setValue("iter_fieldsplit_p_pc_type", "python")
        # Index sets for fieldsplit assignment
        is_YU = self.generate_index_set(W, [0, 1])
        is_P = self.generate_index_set(W, [2])
        # KSP object
        ksp = PETSc.KSP().create(mesh.mpi_comm())
        ksp.setOptionsPrefix("iter_")
        ksp.setFromOptions()
        ksp.getPC().setFieldSplitIS(["yu", is_YU], ["p", is_P])
        ksp.setOperators(A.mat(), AP.mat())
        ksp.setUp()
        ksp_yu, ksp_p = ksp.getPC().getFieldSplitSubKSP()
        ksp_p.pc.setPythonContext(SchurComplementPreconditioner())
        self.ksp = ksp
        
    def solve(self):
        self.ksp.solve(F.vec(), yup.block_vector().vec())
        yup.block_vector().apply("")
        yup.apply("to subfunctions")
        
    @staticmethod
    def generate_index_set(block_function_space, components):
        cpp_code = """
            #include <pybind11/pybind11.h>
            #include <pybind11/stl.h>
            #include <multiphenics/function/BlockFunctionSpace.h>
            
            std::vector<dolfin::la_index> block_owned_dofs__global_numbering(std::shared_ptr<multiphenics::BlockFunctionSpace> block_function_space, std::size_t component)
            {
                return block_function_space->block_dofmap()->block_owned_dofs__global_numbering(component);
            }
            
            PYBIND11_MODULE(SIGNATURE, m)
            {
                m.def("block_owned_dofs__global_numbering", &block_owned_dofs__global_numbering);
            }
        """
        multiphenics_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(multiphenics.__file__)), ".."))
        cpp_module = compile_cpp_code(cpp_code, include_dirs=[multiphenics_dir])
        dofs = list()
        for component in components:
            dofs.extend(cpp_module.block_owned_dofs__global_numbering(block_function_space._cpp_object, component))
        return PETSc.IS().createGeneral(dofs, mesh.mpi_comm())
        
# Iterative solve
yup.block_vector().zero()
yup.block_vector().apply("")
IterativeSolver().solve()
print("Optimal J (iterative) =", assemble(J))
assert isclose(assemble(J), 0.03942770114286462)
plt.figure()
plot(y, title="state")
plt.figure()
plot(u, title="control")
plt.figure()
plot(p, title="adjoint")
plt.show()

# Direct solver class
class DirectSolver(object):
    def __init__(self):
        # PETSc options for direct solvers
        direct_opt = PETSc.Options()
        direct_opt.setValue("direct_ksp_type", "preonly")
        direct_opt.setValue("direct_pc_type", "lu")
        if mesh.mpi_comm().size is 1:
            direct_opt.setValue("direct_pc_factor_mat_solver_package", "umfpack")
        else:
            direct_opt.setValue("direct_pc_factor_mat_solver_package", "pastix") # TODO mumps fails (yet, it works on an equivalent dolfin code!)
        # KSP object
        ksp = PETSc.KSP().create(mesh.mpi_comm())
        ksp.setOptionsPrefix("direct_")
        ksp.setFromOptions()
        ksp.setOperators(A.mat())
        ksp.setUp()
        self.ksp = ksp
        
    def solve(self):
        self.ksp.solve(F.vec(), yup.block_vector().vec())
        yup.block_vector().apply("")
        yup.apply("to subfunctions")

# Direct solve
yup.block_vector().zero()
yup.block_vector().apply("")
DirectSolver().solve()
print("Optimal J (direct) =", assemble(J))
assert isclose(assemble(J), 0.03942770190115938)
plt.figure()
plot(y, title="state")
plt.figure()
plot(u, title="control")
plt.figure()
plot(p, title="adjoint")
plt.show()
