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

import numbers
import pytest
from _pytest.mark import ParameterSet
from numpy import allclose as float_array_equal, array_equal as integer_array_equal, bmat, full, hstack, hstack as bvec, logical_and, sort, unique, vstack
from scipy.sparse import csr_matrix
from dolfin import as_matrix, as_tensor, as_vector, DOLFIN_EPS, dx, FiniteElement, Function, FunctionSpace, inner, MeshFunction, MixedElement, project, SpatialCoordinate, SubDomain, TensorElement, TensorFunctionSpace, VectorElement, VectorFunctionSpace
from dolfin.cpp.la import PETScMatrix, PETScVector
from dolfin.fem import assemble
from multiphenics import BlockDirichletBC, BlockFunction, block_split, BlockTestFunction, BlockTrialFunction, DirichletBC
from multiphenics.fem import block_assemble, BlockDirichletBCLegacy, DirichletBCLegacy

# ================ PYTEST HELPER ================ #
def pytest_mark_slow(item):
    return pytest.param(item, marks=pytest.mark.slow)

def pytest_mark_slow_for_cartesian_product(generator_1, generator_2):
    for i in generator_1():
        broken_i = False # TODO remove when pytest_mark_broken_by_dolfinx is removed
        slow_i = False
        if isinstance(i, ParameterSet):
            assert len(i.marks) is 1
            if i.marks[0].name == "skip": # TODO remove when pytest_mark_broken_by_dolfinx is removed
                assert len(i.values) is 1
                i = i.values[0]
                broken_i = True
            else:
                assert i.marks[0].name == "slow"
                assert len(i.values) is 1
                i = i.values[0]
                slow_i = True
        for j in generator_2():
            broken_j = False # TODO remove when pytest_mark_broken_by_dolfinx is removed
            slow_j = False
            if isinstance(j, ParameterSet):
                assert len(j.marks) is 1
                if j.marks[0].name == "skip": # TODO remove when pytest_mark_broken_by_dolfinx is removed
                    assert len(j.values) is 1
                    j = j.values[0]
                    broken_j = True
                else:
                    assert j.marks[0].name == "slow"
                    assert len(j.values) is 1
                    j = j.values[0]
                    slow_j = True
            assert not isinstance(i, ParameterSet)
            assert not isinstance(j, ParameterSet)
            if broken_i or broken_j: # TODO remove when pytest_mark_broken_by_dolfinx is removed
                yield pytest_mark_broken_by_dolfinx((i, j), -1)
            elif slow_i or slow_j:
                yield pytest_mark_slow((i, j))
            else:
                yield (i, j)
                
def pytest_mark_broken_by_dolfinx(item, dolfinx_issue_number): # TODO remove when dolfinx issue is fixed
    if isinstance(item, ParameterSet):
        assert len(item.marks) is 1
        assert item.marks[0].name == "slow"
        assert len(item.values) is 1
        item = item.values[0]
    return pytest.param(item, marks=pytest.mark.skip)

# ================ EQUALITY BETWEEN ARRAYS ================ #
# Floating point equality check
def array_equal(array1, array2):
    if isinstance(array1.dtype, numbers.Integral) and isinstance(array2.dtype, numbers.Integral):
        return len(array1) == len(array2) and integer_array_equal(array1, array2)
    else:
        return len(array1) == len(array2) and float_array_equal(array1, array2)

# This function is required because ordering of dofs is different between dolfin and block libraries
def array_sorted_equal(array1, array2):
    return array_equal(sort(array1), sort(array2))
    
# This function is required because ordering of dofs is different between dolfin and block libraries,
# and because unique elements must be extracted when comparing tensors on subdomains.
def array_unique_equal(array1, array2):
    return array_equal(unique(array1), unique(array2))
    
# ================ EQUALITY BETWEEN DOFS ================ #
def assert_owned_local_dofs(owned_local_dofs, block_owned_local_dofs):
    assert array_sorted_equal(owned_local_dofs, block_owned_local_dofs)
    
def assert_unowned_local_dofs(unowned_local_dofs, block_unowned_local_dofs):
    # Numbering of unowned dofs may be different, we can only check that the size
    # of the two vectors are consistent
    assert len(unowned_local_dofs) == len(block_unowned_local_dofs)
    
def assert_global_dofs(global_dofs, block_global_dofs):
    assert array_sorted_equal(global_dofs, block_global_dofs)
    
def assert_tabulated_dof_coordinates(dof_coordinates, block_dof_coordinates):
    assert array_equal(dof_coordinates, block_dof_coordinates)
    
# ================ EQUALITY BETWEEN BLOCK VECTORS ================ #
def assert_block_vectors_equal(rhs, block_rhs, block_V):
    if isinstance(rhs, tuple):
        rhs1 = rhs[0]
        rhs2 = rhs[1]
    else:
        rhs1 = rhs
        rhs2 = None
    comm = block_rhs.vec().getComm().tompi4py()
    if rhs2 is not None:
        map_block_to_original = allgather((block_V.block_dofmap().block_to_original(0), block_V.block_dofmap().block_to_original(1)), comm, block_dofmap=block_V.block_dofmap(), dofmap=(block_V[0].dofmap(), block_V[1].dofmap()))
        rhs1g = allgather(rhs1, comm)
        rhs2g = allgather(rhs2, comm)
        rhsg = bvec([rhs1g, rhs2g])
    else:
        map_block_to_original = allgather(block_V.block_dofmap().block_to_original(0), comm, block_dofmap=block_V.block_dofmap(), dofmap=block_V[0].dofmap())
        rhs1g = allgather(rhs1, comm)
        rhsg = rhs1g
    block_rhsg = allgather(block_rhs, comm)
    assert block_rhsg.shape[0] == len(map_block_to_original)
    rhsg_for_assert = block_rhsg*0.
    for (block, original) in map_block_to_original.items():
        rhsg_for_assert[block] = rhsg[original]
    assert array_equal(rhsg_for_assert, block_rhsg)
    
# ================ EQUALITY BETWEEN BLOCK MATRICES ================ #
def assert_block_matrices_equal(lhs, block_lhs, block_V):
    if isinstance(lhs, tuple):
        lhs11 = lhs[0][0]
        lhs12 = lhs[0][1]
        lhs21 = lhs[1][0]
        lhs22 = lhs[1][1]
    else:
        lhs11 = lhs
        lhs12 = None
        lhs21 = None
        lhs22 = None
    comm = block_lhs.mat().getComm().tompi4py()
    if lhs22 is not None:
        map_block_to_original = allgather((block_V.block_dofmap().block_to_original(0), block_V.block_dofmap().block_to_original(1)), comm, block_dofmap=block_V.block_dofmap(), dofmap=(block_V[0].dofmap(), block_V[1].dofmap()))
        lhs11g = allgather(lhs11, comm)
        lhs12g = allgather(lhs12, comm)
        lhs21g = allgather(lhs21, comm)
        lhs22g = allgather(lhs22, comm)
        lhsg = bmat([[lhs11g, lhs12g], [lhs21g, lhs22g]])
    else:
        map_block_to_original = allgather(block_V.block_dofmap().block_to_original(0), comm, block_dofmap=block_V.block_dofmap(), dofmap=block_V[0].dofmap())
        lhs11g = allgather(lhs11, comm)
        lhsg = lhs11g
    block_lhsg = allgather(block_lhs, comm)
    assert block_lhsg.shape[0] == len(map_block_to_original)
    assert block_lhsg.shape[1] == len(map_block_to_original)
    lhsg_for_assert = block_lhsg*0.
    for (block_i, original_i) in map_block_to_original.items():
        for (block_j, original_j) in map_block_to_original.items():
            lhsg_for_assert[block_i, block_j] = lhsg[original_i, original_j]
    assert array_equal(lhsg_for_assert, block_lhsg)
    
# ================ EQUALITY BETWEEN BLOCK FUNCTIONS ================ #
def assert_block_functions_equal(functions, block_function, block_V):
    if functions is None and block_function is None:
        pass
    elif isinstance(functions, tuple):
        assert_block_vectors_equal((functions[0].vector(), functions[1].vector()), block_function.block_vector(), block_V)
    else:
        assert_block_vectors_equal(functions.vector(), block_function.block_vector(), block_V)
    
def assert_functions_manipulations(functions, block_V):
    n_blocks = len(functions)
    assert n_blocks in (1, 2)
    # a) Convert from a list of Functions to a BlockFunction
    block_function_a = BlockFunction(block_V)
    for (index, function) in enumerate(functions):
        function.vector().vec().copy(result=block_function_a.sub(index).vector().vec())
        block_function_a.sub(index).vector().apply()
    block_function_a.apply("from subfunctions")
    # Block vector should have received the data stored in the list of Functions
    if n_blocks == 1:
        assert_block_functions_equal(functions[0], block_function_a, block_V)
    else:
        assert_block_functions_equal((functions[0], functions[1]), block_function_a, block_V)
    # Clean up non-zero values in the restriction by clearing subfunctions and reassigning
    # their values (only on restrictions) from the block_vector. This is not needed in general,
    # but it is required in order to simplify the test b).
    for index in range(n_blocks):
        block_function_a.sub(index).vector().set(0.)
    block_function_a.apply("to subfunctions")
    # b) Test assignment of BlockFunctions
    block_function_b = BlockFunction(block_V)
    block_function_a.block_vector().vec().copy(result=block_function_b.block_vector().vec())
    block_function_b.block_vector().apply()
    block_function_b.apply("to subfunctions")
    # Each sub function should now contain the same data as the original block function
    for index in range(n_blocks):
        assert array_equal(block_function_b.sub(index).vector().get_local(), block_function_a.sub(index).vector().get_local())
    # The two block vectors should store the same data
    assert array_equal(block_function_b.block_vector().get_local(), block_function_a.block_vector().get_local())
    
# ================ FUNCTION SPACES GENERATOR ================ #
def StokesFunctionSpace(mesh, family_degree):
    stokes_element = StokesElement(family_degree[0], mesh.ufl_cell(), family_degree[1])
    return FunctionSpace(mesh, stokes_element)

def StokesElement(family, cell, degree):
    V_element = VectorElement(family, cell, degree + 1)
    Q_element = FiniteElement(family, cell, degree)
    return MixedElement(V_element, Q_element)
    
def FunctionAndRealSpace(mesh, family_degree):
    function_and_real_element = FunctionAndRealElement(family_degree[0], mesh.ufl_cell(), family_degree[1])
    return FunctionSpace(mesh, function_and_real_element)

def FunctionAndRealElement(family, cell, degree):
    V_element = FiniteElement(family, cell, degree)
    R_element = FiniteElement("Real", cell, 0)
    return MixedElement(V_element, R_element)
    
def get_function_spaces_1():
    return (
        lambda mesh: FunctionSpace(mesh, ("Lagrange", 1)),
        pytest_mark_slow(lambda mesh: FunctionSpace(mesh, ("Lagrange", 2))),
        lambda mesh: VectorFunctionSpace(mesh, ("Lagrange", 1)),
        pytest_mark_slow(lambda mesh: VectorFunctionSpace(mesh, ("Lagrange", 2))),
        pytest_mark_slow(lambda mesh: TensorFunctionSpace(mesh, ("Lagrange", 1))),
        pytest_mark_slow(lambda mesh: TensorFunctionSpace(mesh, ("Lagrange", 2))),
        lambda mesh: StokesFunctionSpace(mesh, ("Lagrange", 1)),
        pytest_mark_slow(lambda mesh: StokesFunctionSpace(mesh, ("Lagrange", 2))),
        pytest_mark_broken_by_dolfinx(lambda mesh: FunctionSpace(mesh, ("Real", 0)), 225),
        pytest_mark_broken_by_dolfinx(pytest_mark_slow(lambda mesh: VectorFunctionSpace(mesh, ("Real", 0))), 225),
        pytest_mark_broken_by_dolfinx(pytest_mark_slow(lambda mesh: FunctionAndRealSpace(mesh, ("Lagrange", 1))), 225),
        pytest_mark_broken_by_dolfinx(pytest_mark_slow(lambda mesh: FunctionAndRealSpace(mesh, ("Lagrange", 2))), 225)
    )
    
def get_function_spaces_2():
    return pytest_mark_slow_for_cartesian_product(get_function_spaces_1, get_function_spaces_1)
    
def get_elements_1():
    return (
        lambda mesh: FiniteElement("Lagrange", mesh.ufl_cell(), 1),
        pytest_mark_slow(lambda mesh: FiniteElement("Lagrange", mesh.ufl_cell(), 2)),
        lambda mesh: VectorElement("Lagrange", mesh.ufl_cell(), 1),
        pytest_mark_slow(lambda mesh: VectorElement("Lagrange", mesh.ufl_cell(), 2)),
        pytest_mark_slow(lambda mesh: TensorElement("Lagrange", mesh.ufl_cell(), 1)),
        pytest_mark_slow(lambda mesh: TensorElement("Lagrange", mesh.ufl_cell(), 2)),
        lambda mesh: StokesElement("Lagrange", mesh.ufl_cell(), 1),
        pytest_mark_slow(lambda mesh: StokesElement("Lagrange", mesh.ufl_cell(), 2)),
        pytest_mark_broken_by_dolfinx(lambda mesh: FiniteElement("Real", mesh.ufl_cell(), 0), 225),
        pytest_mark_broken_by_dolfinx(pytest_mark_slow(lambda mesh: VectorElement("Real", mesh.ufl_cell(), 0)), 225),
        pytest_mark_broken_by_dolfinx(pytest_mark_slow(lambda mesh: FunctionAndRealElement("Lagrange", mesh.ufl_cell(), 1)), 225),
        pytest_mark_broken_by_dolfinx(pytest_mark_slow(lambda mesh: FunctionAndRealElement("Lagrange", mesh.ufl_cell(), 2)), 225)
    )
    
def get_elements_2():
    return pytest_mark_slow_for_cartesian_product(get_elements_1, get_elements_1)
    
# ================ SUBDOMAIN GENERATOR ================ #
def UnitSquareSubDomain(X, Y):
    class CustomSubDomain(SubDomain):
        def inside(self, x, on_boundary):
            return logical_and(x[:, 0] <= X, x[:, 1] <= Y)
    return CustomSubDomain()

def UnitSquareInterface(X=None, Y=None, on_boundary=False):
    assert (
        (X is not None and Y is None and on_boundary is False)
            or
        (X is None and Y is not None and on_boundary is False)
            or
        (X is None and Y is None and on_boundary is True)
    )
    if X is not None:
        class CustomSubDomain(SubDomain):
            def inside(self, x, on_boundary_):
                return logical_and(x[:, 0] >= X - DOLFIN_EPS, x[:, 0] <= X + DOLFIN_EPS)
    elif Y is not None:
        class CustomSubDomain(SubDomain):
            def inside(self, x, on_boundary_):
                return logical_and(x[:, 1] >= Y - DOLFIN_EPS, x[:, 1] <= Y + DOLFIN_EPS)
    elif on_boundary is True:
        class CustomSubDomain(SubDomain):
            def inside(self, x, on_boundary_):
                return full(x.shape[0], on_boundary_)
    return CustomSubDomain()
    
def OnBoundary():
    return UnitSquareInterface(on_boundary=True)
    
def get_restrictions_1():
    return (
        None,
        UnitSquareSubDomain(0.5, 0.5),
        UnitSquareInterface(on_boundary=True),
        pytest_mark_slow(UnitSquareInterface(X=1.0)),
        pytest_mark_slow(UnitSquareInterface(Y=0.0)),
        UnitSquareInterface(X=0.75),
        pytest_mark_slow(UnitSquareInterface(Y=0.25))
    )

def get_restrictions_2():
    return (
        (None, None),
        (None, UnitSquareSubDomain(0.75, 0.75)),
        pytest_mark_slow((None, UnitSquareInterface(on_boundary=True))),
        (None, UnitSquareInterface(Y=0.0)),
        pytest_mark_slow((UnitSquareSubDomain(0.5, 0.75), None)),
        pytest_mark_slow((UnitSquareInterface(on_boundary=True), None)),
        pytest_mark_slow((UnitSquareInterface(X=1.0), None)),
        (UnitSquareSubDomain(0.75, 0.75), UnitSquareSubDomain(0.75, 0.75)),
        pytest_mark_slow((UnitSquareSubDomain(0.5, 0.75), UnitSquareSubDomain(0.75, 0.75))),
        pytest_mark_slow((UnitSquareInterface(on_boundary=True), UnitSquareInterface(on_boundary=True))),
        (UnitSquareInterface(on_boundary=True), UnitSquareInterface(X=1.0)),
        pytest_mark_slow((UnitSquareInterface(X=1.0), UnitSquareInterface(on_boundary=True))),
        (UnitSquareInterface(X=1.0), UnitSquareInterface(Y=0.0)),
        pytest_mark_slow((UnitSquareInterface(X=0.75), UnitSquareInterface(Y=0.0))),
        pytest_mark_slow((UnitSquareInterface(X=0.75), UnitSquareInterface(Y=0.25))),
        (UnitSquareSubDomain(0.5, 0.75), UnitSquareInterface(on_boundary=True)),
        pytest_mark_slow((UnitSquareSubDomain(0.5, 0.75), UnitSquareInterface(Y=0.25))),
        pytest_mark_slow((UnitSquareInterface(on_boundary=True), UnitSquareSubDomain(0.5, 0.75))),
        pytest_mark_slow((UnitSquareInterface(Y=0.25), UnitSquareSubDomain(0.5, 0.75)))
    )
    
# ================ BLOCK BOUNDARY CONDITIONS GENERATOR ================ #
# Computation of block bcs for single block
def get_block_bcs_1():
    def _get_bc_1(block_V):
        mesh = block_V.mesh()
        boundaries = MeshFunction("size_t", mesh, mesh.topology.dim - 1, 0)
        OnBoundary().mark(boundaries, 1)
        num_sub_elements = block_V[0].ufl_element().num_sub_elements()
        if num_sub_elements is 0:
            bc1_fun = Function(block_V[0])
            bc1_fun.vector().set(1.)
            return [DirichletBC(block_V[0], bc1_fun, (boundaries, 1))]
        else:
            bc1 = list()
            for i in range(num_sub_elements):
                bc1_fun = Function(block_V[0].sub(i).collapse())
                bc1_fun.vector().set(i + 1.)
                bc1.append(DirichletBC(block_V[0].sub(i), bc1_fun, (boundaries, 1)))
            return bc1
    return (
        lambda block_V: None,
        pytest_mark_slow(lambda block_V: BlockDirichletBC([None], block_function_space=block_V)),
        lambda block_V: BlockDirichletBC(_get_bc_1(block_V))
    )
    
# Computation of block bcs for two blocks
def get_block_bcs_2():
    def _get_bc_1(block_V):
        mesh = block_V.mesh()
        boundaries = MeshFunction("size_t", mesh, mesh.topology.dim - 1, 0)
        OnBoundary().mark(boundaries, 1)
        num_sub_elements = block_V[0].ufl_element().num_sub_elements()
        if num_sub_elements is 0:
            bc1_fun = Function(block_V[0])
            bc1_fun.vector().set(1.)
            return [DirichletBC(block_V[0], bc1_fun, (boundaries, 1))]
        else:
            bc1 = list()
            for i in range(num_sub_elements):
                bc1_fun = Function(block_V[0].sub(i).collapse())
                bc1_fun.vector().set(i + 1.)
                bc1.append(DirichletBC(block_V[0].sub(i), bc1_fun, (boundaries, 1)))
            return bc1
    def _get_bc_2(block_V):
        mesh = block_V.mesh()
        boundaries = MeshFunction("size_t", mesh, mesh.topology.dim - 1, 0)
        OnBoundary().mark(boundaries, 1)
        num_sub_elements = block_V[1].ufl_element().num_sub_elements()
        if num_sub_elements is 0:
            bc2_fun = Function(block_V[1])
            bc2_fun.vector().set(11.)
            return [DirichletBC(block_V[1], bc2_fun, (boundaries, 1))]
        else:
            bc2 = list()
            for i in range(num_sub_elements):
                bc2_fun = Function(block_V[1].sub(i).collapse())
                bc2_fun.vector().set(i + 11.)
                bc2.append(DirichletBC(block_V[1].sub(i), bc2_fun, (boundaries, 1)))
            return bc2
    return (
        lambda block_V: None,
        pytest_mark_slow(lambda block_V: BlockDirichletBC([None, None], block_function_space=block_V)),
        lambda block_V: BlockDirichletBC([_get_bc_1(block_V), None]),
        pytest_mark_slow(lambda block_V: BlockDirichletBC([None, _get_bc_2(block_V)])),
        lambda block_V: BlockDirichletBC([_get_bc_1(block_V), _get_bc_2(block_V)])
    )
    
# ================ RIGHT-HAND SIDE BLOCK FORM GENERATOR ================ #
# Computation of rhs block form for single block
def get_rhs_block_form_1(block_V):
    block_v = BlockTestFunction(block_V)
    (v, ) = block_split(block_v)
    x = SpatialCoordinate(block_V.mesh())
    shape_1 = block_V[0].ufl_element().value_shape()
    if len(shape_1) is 0:
        f = 2*x[0] + 4*x[1]*x[1]
        block_form = [f*v*dx]
    elif len(shape_1) is 1 and shape_1[0] is 2:
        f = as_vector((2*x[0] + 4*x[1]*x[1], 3*x[0] + 5*x[1]*x[1]))
        block_form = [inner(f, v)*dx]
    elif len(shape_1) is 1 and shape_1[0] is 3:
        f = as_vector((2*x[0] + 4*x[1]*x[1], 3*x[0] + 5*x[1]*x[1], 7*x[0] + 11*x[1]*x[1]))
        block_form = [inner(f, v)*dx]
    elif len(shape_1) is 2:
        f = as_matrix(((2*x[0] + 4*x[1]*x[1], 3*x[0] + 5*x[1]*x[1]),
                       (7*x[0] + 11*x[1]*x[1], 13*x[0] + 17*x[1]*x[1])))
        block_form = [inner(f, v)*dx]
    return block_form
    
# Computation of rhs block form for two blocks
def get_rhs_block_form_2(block_V):
    block_v = BlockTestFunction(block_V)
    (v1, v2) = block_split(block_v)
    x = SpatialCoordinate(block_V.mesh())
    block_form = [None, None]
    shape_1 = block_V[0].ufl_element().value_shape()
    if len(shape_1) is 0:
        f1 = 2*x[0] + 4*x[1]*x[1]
        block_form[0] = f1*v1*dx
    elif len(shape_1) is 1 and shape_1[0] is 2:
        f1 = as_vector((2*x[0] + 4*x[1]*x[1], 3*x[0] + 5*x[1]*x[1]))
        block_form[0] = inner(f1, v1)*dx
    elif len(shape_1) is 1 and shape_1[0] is 3:
        f1 = as_vector((2*x[0] + 4*x[1]*x[1], 3*x[0] + 5*x[1]*x[1], 7*x[0] + 11*x[1]*x[1]))
        block_form[0] = inner(f1, v1)*dx
    elif len(shape_1) is 2:
        f1 = as_matrix(((2*x[0] + 4*x[1]*x[1], 3*x[0] + 5*x[1]*x[1]),
                        (7*x[0] + 11*x[1]*x[1], 13*x[0] + 17*x[1]*x[1])))
        block_form[0] = inner(f1, v1)*dx
    shape_2 = block_V[1].ufl_element().value_shape()
    if len(shape_2) is 0:
        f2 = 2*x[1] + 4*x[0]*x[0]
        block_form[1] = f2*v2*dx
    elif len(shape_2) is 1 and shape_2[0] is 2:
        f2 = as_vector((2*x[1] + 4*x[0]*x[0], 3*x[1] + 5*x[0]*x[0]))
        block_form[1] = inner(f2, v2)*dx
    elif len(shape_2) is 1 and shape_2[0] is 3:
        f2 = as_vector((2*x[1] + 4*x[0]*x[0], 3*x[1] + 5*x[0]*x[0], 7*x[1] + 11*x[0]*x[0]))
        block_form[1] = inner(f2, v2)*dx
    elif len(shape_2) is 2:
        f2 = as_matrix(((2*x[1] + 4*x[0]*x[0], 3*x[1] + 5*x[0]*x[0]),
                        (7*x[1] + 11*x[0]*x[0], 13*x[1] + 17*x[0]*x[0])))
        block_form[1] = inner(f2, v2)*dx
    return block_form
    
# ================ LEFT-HAND SIDE BLOCK FORM GENERATOR ================ #
# Computation of lhs block form for single block
def get_lhs_block_form_1(block_V):
    block_u = BlockTrialFunction(block_V)
    block_v = BlockTestFunction(block_V)
    (u, ) = block_split(block_u)
    (v, ) = block_split(block_v)
    x = SpatialCoordinate(block_V.mesh())
    shape_1 = block_V[0].ufl_element().value_shape()
    if len(shape_1) is 0:
        f = 2*x[0] + 4*x[1]*x[1]
        block_form = [[f*u*v*dx]]
    elif len(shape_1) is 1 and shape_1[0] is 2:
        f = as_vector((2*x[0] + 4*x[1]*x[1], 3*x[0] + 5*x[1]*x[1]))
        block_form = [[(f[0]*u[0]*v[0] + f[1]*u[1].dx(1)*v[1])*dx]]
    elif len(shape_1) is 1 and shape_1[0] is 3:
        f = as_vector((2*x[0] + 4*x[1]*x[1], 3*x[0] + 5*x[1]*x[1], 7*x[0] + 11*x[1]*x[1]))
        block_form = [[(f[0]*u[0]*v[0] + f[1]*u[1].dx(1)*v[1] + f[2]*u[2].dx(0)*v[2].dx(1))*dx]]
    elif len(shape_1) is 2:
        f = as_tensor(((2*x[0] + 4*x[1]*x[1], 3*x[0] + 5*x[1]*x[1]),
                       (7*x[0] + 11*x[1]*x[1], 13*x[0] + 17*x[1]*x[1])))
        block_form = [[(f[0, 0]*u[0, 0]*v[0, 0] + f[0, 1]*u[0, 1].dx(1)*v[0, 1] + f[1, 0]*u[1, 0].dx(0)*v[1, 0].dx(1) + f[1, 1]*u[1, 1].dx(0)*v[1, 1])*dx]]
    return block_form
    
# Computation of lhs block form for two blocks
def get_lhs_block_form_2(block_V):
    block_u = BlockTrialFunction(block_V)
    block_v = BlockTestFunction(block_V)
    (u1, u2) = block_split(block_u)
    (v1, v2) = block_split(block_v)
    x = SpatialCoordinate(block_V.mesh())
    block_form = [[None, None], [None, None]]
    # (1, 1) block
    shape_1 = block_V[0].ufl_element().value_shape()
    if len(shape_1) is 0:
        f1 = 2*x[0] + 4*x[1]*x[1]
        block_form[0][0] = f1*u1*v1*dx
    elif len(shape_1) is 1 and shape_1[0] is 2:
        f1 = as_vector((2*x[0] + 4*x[1]*x[1], 3*x[0] + 5*x[1]*x[1]))
        block_form[0][0] = (f1[0]*u1[0]*v1[0] + f1[1]*u1[1].dx(1)*v1[1])*dx
    elif len(shape_1) is 1 and shape_1[0] is 3:
        f1 = as_vector((2*x[0] + 4*x[1]*x[1], 3*x[0] + 5*x[1]*x[1], 7*x[0] + 11*x[1]*x[1]))
        block_form[0][0] = (f1[0]*u1[0]*v1[0] + f1[1]*u1[1].dx(1)*v1[1] + f1[2]*u1[2].dx(0)*v1[2].dx(1))*dx
    elif len(shape_1) is 2:
        f1 = as_matrix(((2*x[0] + 4*x[1]*x[1], 3*x[0] + 5*x[1]*x[1]),
                        (7*x[0] + 11*x[1]*x[1], 13*x[0] + 17*x[1]*x[1])))
        block_form[0][0] = (f1[0, 0]*u1[0, 0]*v1[0, 0] + f1[0, 1]*u1[0, 1].dx(1)*v1[0, 1] + f1[1, 0]*u1[1, 0].dx(0)*v1[1, 0].dx(1) + f1[1, 1]*u1[1, 1].dx(0)*v1[1, 1])*dx
    # (2, 2) block
    shape_2 = block_V[1].ufl_element().value_shape()
    if len(shape_2) is 0:
        f2 = 2*x[1] + 4*x[0]*x[0]
        block_form[1][1] = f2*u2*v2*dx
    elif len(shape_2) is 1 and shape_2[0] is 2:
        f2 = as_vector((2*x[1] + 4*x[0]*x[0], 3*x[1] + 5*x[0]*x[0]))
        block_form[1][1] = (f2[0]*u2[0]*v2[0] + f2[1]*u2[1].dx(1)*v2[1])*dx
    elif len(shape_2) is 1 and shape_2[0] is 3:
        f2 = as_vector((2*x[1] + 4*x[0]*x[0], 3*x[1] + 5*x[0]*x[0], 7*x[1] + 11*x[0]*x[0]))
        block_form[1][1] = (f2[0]*u2[0]*v2[0] + f2[1]*u2[1].dx(1)*v2[1] + f2[2]*u2[2].dx(0)*v2[2].dx(1))*dx
    elif len(shape_2) is 2:
        f2 = as_matrix(((2*x[1] + 4*x[0]*x[0], 3*x[1] + 5*x[0]*x[0]),
                        (7*x[1] + 11*x[0]*x[0], 13*x[1] + 17*x[0]*x[0])))
        block_form[1][1] = (f2[0, 0]*u2[0, 0]*v2[0, 0] + f2[0, 1]*u2[0, 1].dx(1)*v2[0, 1] + f2[1, 0]*u2[1, 0].dx(0)*v2[1, 0].dx(1) + f2[1, 1]*u2[1, 1].dx(0)*v2[1, 1])*dx
    # (1, 2) and (2, 1) blocks
    if len(shape_1) is 0:
        if len(shape_2) is 0:
            block_form[0][1] = f1*u2*v1*dx
            block_form[1][0] = f2*u1*v2*dx
        elif len(shape_2) is 1 and shape_2[0] is 2:
            block_form[0][1] = f1*u2[0]*v1*dx + f1*u2[1]*v1.dx(1)*dx
            block_form[1][0] = (f2[0]*u1*v2[0] + f2[1]*u1.dx(1)*v2[1])*dx
        elif len(shape_2) is 1 and shape_2[0] is 3:
            block_form[0][1] = f1*u2[0]*v1*dx + f1*u2[1]*v1.dx(1)*dx + f1*u2[2]*v1*dx
            block_form[1][0] = (f2[0]*u1*v2[0] + f2[1]*u1.dx(1)*v2[1] + f2[2]*u1.dx(0)*v2[2].dx(1))*dx
        elif len(shape_2) is 2:
            block_form[0][1] = f1*u2[0, 0]*v1*dx + f1*u2[1, 1]*v1.dx(0)*dx
            block_form[1][0] = (f2[0, 0]*u1*v2[0, 0] + f2[0, 1]*u1.dx(1)*v2[0, 1] + f2[1, 0]*u1.dx(0)*v2[1, 0].dx(1) + f2[1, 1]*u1.dx(0)*v2[1, 1])*dx
    elif len(shape_1) is 1 and shape_1[0] is 2:
        if len(shape_2) is 0:
            block_form[0][1] = (f1[0]*u2*v1[0] + f1[1]*u2.dx(1)*v1[1])*dx
            block_form[1][0] = f2*u1[0]*v2*dx + f2*u1[1]*v2.dx(0)*dx
        elif len(shape_2) is 1 and shape_2[0] is 2:
            block_form[0][1] = (f1[0]*u2[0]*v1[0] + f1[1]*u2[1].dx(1)*v1[1])*dx
            block_form[1][0] = (f2[0]*u1[0]*v2[0] + f2[1]*u1[1].dx(1)*v2[1])*dx
        elif len(shape_2) is 1 and shape_2[0] is 3:
            block_form[0][1] = (f1[0]*u2[0]*v1[0] + f1[1]*u2[1].dx(1)*v1[1] + f1[0]*u2[2]*v1[0])*dx
            block_form[1][0] = (f2[0]*u1[0]*v2[0] + f2[1]*u1[1].dx(1)*v2[1] + f2[2]*u1[0].dx(0)*v2[2].dx(1))*dx
        elif len(shape_2) is 2:
            block_form[0][1] = (f1[0]*u2[0, 0]*v1[0] + f1[1]*u2[1, 1].dx(1)*v1[1])*dx
            block_form[1][0] = (f2[0, 0]*u1[0]*v2[0, 0] + f2[0, 1]*u1[0].dx(1)*v2[0, 1] + f2[1, 0]*u1[1].dx(0)*v2[1, 0].dx(1) + f2[1, 1]*u1[0].dx(0)*v2[1, 1])*dx
    elif len(shape_1) is 1 and shape_1[0] is 3:
        if len(shape_2) is 0:
            block_form[0][1] = (f1[0]*u2*v1[0] + f1[1]*u2.dx(1)*v1[1] + f1[2]*u2.dx(0)*v1[2].dx(1))*dx
            block_form[1][0] = f2*u1[0]*v2*dx + f2*u1[1]*v2.dx(1)*dx + f2*u1[2]*v2*dx
        elif len(shape_2) is 1 and shape_2[0] is 2:
            block_form[0][1] = (f1[0]*u2[0]*v1[0] + f1[1]*u2[1].dx(1)*v1[1] + f1[2]*u2[0].dx(0)*v1[2].dx(1))*dx
            block_form[1][0] = (f2[0]*u1[0]*v2[0] + f2[1]*u1[1].dx(1)*v2[1] + f2[1]*u1[2].dx(1)*v2[1])*dx
        elif len(shape_2) is 1 and shape_2[0] is 3:
            block_form[0][1] = (f1[0]*u2[0]*v1[0] + f1[1]*u2[1].dx(1)*v1[1] + f1[2]*u2[2].dx(0)*v1[2].dx(1))*dx
            block_form[1][0] = (f2[0]*u1[0]*v2[0] + f2[1]*u1[1].dx(1)*v2[1] + f2[2]*u1[2].dx(0)*v2[2].dx(1))*dx
        elif len(shape_2) is 2:
            block_form[0][1] = (f1[0]*u2[0, 0]*v1[0] + f1[1]*u2[1, 0].dx(1)*v1[1] + f1[2]*u2[0, 1].dx(0)*v1[2].dx(1) + f1[0]*u2[1, 1]*v1[0].dx(1))*dx
            block_form[1][0] = (f2[0, 0]*u1[0]*v2[0, 0] + f2[0, 1]*u1[1].dx(1)*v2[0, 1] + f2[1, 0]*u1[2].dx(0)*v2[1, 0].dx(1) + f2[1, 1]*u1[0].dx(0)*v2[1, 1])*dx
    elif len(shape_1) is 2:
        if len(shape_2) is 0:
            block_form[0][1] = (f1[0, 0]*u2*v1[0, 0] + f1[0, 1]*u2.dx(1)*v1[0, 1] + f1[1, 0]*u2.dx(0)*v1[1, 0].dx(1) + f1[1, 1]*u2.dx(0)*v1[1, 1])*dx
            block_form[1][0] = f2*u1[0, 0]*v2*dx + f2*u1[1, 1]*v2.dx(1)*dx
        elif len(shape_2) is 1 and shape_2[0] is 2:
            block_form[0][1] = (f1[0, 0]*u2[0]*v1[0, 0] + f1[0, 1]*u2[0].dx(1)*v1[0, 1] + f1[1, 0]*u2[1].dx(0)*v1[1, 0].dx(1) + f1[1, 1]*u2[1].dx(0)*v1[1, 1])*dx
            block_form[1][0] = (f2[0]*u1[0, 0]*v2[0] + f2[1]*u1[1, 1].dx(1)*v2[1])*dx
        elif len(shape_2) is 1 and shape_2[0] is 3:
            block_form[0][1] = (f1[0, 0]*u2[0]*v1[0, 0] + f1[0, 1]*u2[1].dx(1)*v1[0, 1] + f1[1, 0]*u2[2].dx(0)*v1[1, 0].dx(1) + f1[1, 1]*u2[0].dx(0)*v1[1, 1])*dx
            block_form[1][0] = (f2[0]*u1[0, 0]*v2[0] + f2[1]*u1[1, 0].dx(1)*v2[1] + f2[2]*u1[0, 1].dx(0)*v2[2].dx(1) + f2[0]*u1[1, 1]*v2[0].dx(1))*dx
        elif len(shape_2) is 2:
            block_form[0][1] = (f1[0, 0]*u2[0, 0]*v1[0, 0] + f1[0, 1]*u2[0, 1].dx(1)*v1[0, 1] + f1[1, 0]*u2[1, 0].dx(0)*v1[1, 0].dx(1) + f1[1, 1]*u2[1, 1].dx(0)*v1[1, 1])*dx
            block_form[1][0] = (f2[0, 0]*u1[0, 0]*v2[0, 0] + f2[0, 1]*u1[0, 1].dx(1)*v2[0, 1] + f2[1, 0]*u1[1, 0].dx(0)*v2[1, 0].dx(1) + f2[1, 1]*u1[1, 1].dx(0)*v2[1, 1])*dx
    return block_form
    
# ================ RIGHT-HAND SIDE BLOCK FORM ASSEMBLER ================ #
def assemble_and_block_assemble_vector(block_form):
    N = len(block_form)
    assert N in (1, 2)
    if N == 1:
        return assemble(block_form[0]), block_assemble(block_form)
    else:
        return (assemble(block_form[0]), assemble(block_form[1])), block_assemble(block_form)
        
def apply_bc_and_block_bc_vector(rhs, block_rhs, block_bcs):
    if block_bcs is None:
        return
    N = len(block_bcs)
    assert N in (1, 2)
    if N == 1:
        DirichletBCLegacy.apply(block_bcs[0], rhs)
        BlockDirichletBCLegacy.apply(block_bcs, block_rhs)
    else:
        DirichletBCLegacy.apply(block_bcs[0], rhs[0])
        DirichletBCLegacy.apply(block_bcs[1], rhs[1])
        BlockDirichletBCLegacy.apply(block_bcs, block_rhs)
    
def apply_bc_and_block_bc_vector_non_linear(rhs, block_rhs, block_bcs, block_V):
    if block_bcs is None:
        return (None, None)
    N = len(block_bcs)
    assert N in (1, 2)
    if N == 1:
        function = Function(block_V[0])
        DirichletBCLegacy.apply(block_bcs[0], rhs, function.vector())
        block_function = BlockFunction(block_V)
        BlockDirichletBCLegacy.apply(block_bcs, block_rhs, block_function.block_vector())
        return (function, block_function)
    else:
        function1 = Function(block_V[0])
        DirichletBCLegacy.apply(block_bcs[0], rhs[0], function1.vector())
        function2 = Function(block_V[1])
        DirichletBCLegacy.apply(block_bcs[1], rhs[1], function2.vector())
        block_function = BlockFunction(block_V)
        BlockDirichletBCLegacy.apply(block_bcs, block_rhs, block_function.block_vector())
        return ((function1, function2), block_function)
        
# ================ LEFT-HAND SIDE BLOCK FORM ASSEMBLER ================ #
def assemble_and_block_assemble_matrix(block_form):
    N = len(block_form)
    assert N in (1, 2)
    M = len(block_form[0])
    assert M == N
    if N == 1:
        return assemble(block_form[0][0]), block_assemble(block_form)
    else:
        return ((assemble(block_form[0][0]), assemble(block_form[0][1])), (assemble(block_form[1][0]), assemble(block_form[1][1]))), block_assemble(block_form)
    
def apply_bc_and_block_bc_matrix(lhs, block_lhs, block_bcs):
    if block_bcs is None:
        return
    N = len(block_bcs)
    assert N in (1, 2)
    if N == 1:
        DirichletBCLegacy.apply(block_bcs[0], lhs, 1.)
        BlockDirichletBCLegacy.apply(block_bcs, block_lhs, 1.)
    else:
        DirichletBCLegacy.apply(block_bcs[0], lhs[0][0], 1.)
        DirichletBCLegacy.apply(block_bcs[0], lhs[0][1], 0.)
        DirichletBCLegacy.apply(block_bcs[1], lhs[1][0], 0.)
        DirichletBCLegacy.apply(block_bcs[1], lhs[1][1], 1.)
        BlockDirichletBCLegacy.apply(block_bcs, block_lhs, 1.)
        
# ================ BLOCK FUNCTIONS GENERATOR ================ #
# Computation of block function for single block
def get_list_of_functions_1(block_V):
    x = SpatialCoordinate(block_V.mesh())
    shape_1 = block_V[0].ufl_element().value_shape()
    if len(shape_1) is 0:
        f = 2*x[0] + 4*x[1]*x[1]
    elif len(shape_1) is 1 and shape_1[0] is 2:
        f = as_vector((2*x[0] + 4*x[1]*x[1], 3*x[0] + 5*x[1]*x[1]))
    elif len(shape_1) is 1 and shape_1[0] is 3:
        f = as_vector((2*x[0] + 4*x[1]*x[1], 3*x[0] + 5*x[1]*x[1], 7*x[0] + 11*x[1]*x[1]))
    elif len(shape_1) is 2:
        f = as_matrix(((2*x[0] + 4*x[1]*x[1], 3*x[0] + 5*x[1]*x[1]),
                       (7*x[0] + 11*x[1]*x[1], 13*x[0] + 17*x[1]*x[1])))
    return [project(f, block_V[0])]
    
# Computation of block function for two blocks
def get_list_of_functions_2(block_V):
    x = SpatialCoordinate(block_V.mesh())
    shape_1 = block_V[0].ufl_element().value_shape()
    if len(shape_1) is 0:
        f1 = 2*x[0] + 4*x[1]*x[1]
    elif len(shape_1) is 1 and shape_1[0] is 2:
        f1 = as_vector((2*x[0] + 4*x[1]*x[1], 3*x[0] + 5*x[1]*x[1]))
    elif len(shape_1) is 1 and shape_1[0] is 3:
        f1 = as_vector((2*x[0] + 4*x[1]*x[1], 3*x[0] + 5*x[1]*x[1], 7*x[0] + 11*x[1]*x[1]))
    elif len(shape_1) is 2:
        f1 = as_matrix(((2*x[0] + 4*x[1]*x[1], 3*x[0] + 5*x[1]*x[1]),
                        (7*x[0] + 11*x[1]*x[1], 13*x[0] + 17*x[1]*x[1])))
    shape_2 = block_V[1].ufl_element().value_shape()
    if len(shape_2) is 0:
        f2 = 2*x[1] + 4*x[0]*x[0]
    elif len(shape_2) is 1 and shape_2[0] is 2:
        f2 = as_vector((2*x[1] + 4*x[0]*x[0], 3*x[1] + 5*x[0]*x[0]))
    elif len(shape_2) is 1 and shape_2[0] is 3:
        f2 = as_vector((2*x[1] + 4*x[0]*x[0], 3*x[1] + 5*x[0]*x[0], 7*x[1] + 11*x[0]*x[0]))
    elif len(shape_2) is 2:
        f2 = as_matrix(((2*x[1] + 4*x[0]*x[0], 3*x[1] + 5*x[0]*x[0]),
                        (7*x[1] + 11*x[0]*x[0], 13*x[1] + 17*x[0]*x[0])))
    return [project(f1, block_V[0]), project(f2, block_V[1])]
    
# ================ PARALLEL SUPPORT ================ #
# Gather matrices, vector and dicts on zero-th process
def allgather(obj, comm, **kwargs):
    assert isinstance(obj, (dict, tuple, PETScMatrix, PETScVector))
    if isinstance(obj, (dict, tuple)):
        assert "block_dofmap" in kwargs
        assert "dofmap" in kwargs
        if isinstance(obj, tuple):
            assert isinstance(kwargs["dofmap"], tuple)
            all_block_to_original1 = comm.allgather(obj[0])
            all_ownership_ranges1 = comm.allgather(kwargs["dofmap"][0].ownership_range())
            all_block_ownership_ranges1 = comm.allgather(kwargs["block_dofmap"].sub_index_map(0).local_range())
            all_block_to_original2 = comm.allgather(obj[1])
            all_ownership_ranges2 = comm.allgather(kwargs["dofmap"][1].ownership_range())
            all_block_ownership_ranges2 = comm.allgather(kwargs["block_dofmap"].sub_index_map(1).local_range())
            base_index1 = [None]*comm.Get_size()
            block_base_index1 = [None]*comm.Get_size()
            base_index2 = [None]*comm.Get_size()
            block_base_index2 = [None]*comm.Get_size()
            for r in range(comm.Get_size() + 1):
                if r == 0:
                    base_index1[0] = 0
                    base_index2[0] = all_ownership_ranges1[-1][1]
                    block_base_index1[0] = 0
                if r > 0:
                    block_base_index2[r-1] = block_base_index1[r-1] + (all_block_ownership_ranges1[r-1][1] - all_block_ownership_ranges1[r-1][0])
                    if r < comm.Get_size():
                        base_index1[r] = all_ownership_ranges1[r-1][1]
                        base_index2[r] = all_ownership_ranges1[-1][1] + all_ownership_ranges2[r-1][1]
                        block_base_index1[r] = block_base_index2[r-1] + (all_block_ownership_ranges2[r-1][1] - all_block_ownership_ranges2[r-1][0])
            output = dict()
            for r in range(comm.Get_size()):
                for (block1, original1) in all_block_to_original1[r].items():
                    if original1 < all_ownership_ranges1[r][1] - all_ownership_ranges1[r][0]:
                        output[block1 + block_base_index1[r]] = original1 + base_index1[r]
                for (block2, original2) in all_block_to_original2[r].items():
                    if original2 < all_ownership_ranges2[r][1] - all_ownership_ranges2[r][0]:
                        # Note that we use block_base_index1 instead of block_base_index2 due to internal storage of block2
                        output[block2 + block_base_index1[r]] = original2 + base_index2[r]
            return output
        else:
            assert isinstance(obj, dict)
            all_block_to_original1 = comm.allgather(obj)
            all_ownership_ranges1 = comm.allgather(kwargs["dofmap"].ownership_range())
            all_block_ownership_ranges1 = comm.allgather(kwargs["block_dofmap"].sub_index_map(0).local_range())
            base_index1 = [ownr[0] for ownr in all_ownership_ranges1]
            block_base_index1 = [ownr[0] for ownr in all_block_ownership_ranges1]
            output = dict()
            for r in range(comm.Get_size()):
                for (block1, original1) in all_block_to_original1[r].items():
                    if original1 < all_ownership_ranges1[r][1] - all_ownership_ranges1[r][0]:
                        output[block1 + block_base_index1[r]] = original1 + base_index1[r]
            return output
    elif isinstance(obj, PETScMatrix):
        return vstack(comm.allgather(to_dense(obj)))
    elif isinstance(obj, PETScVector):
        return hstack(comm.allgather(to_dense(obj)))
    else:
        raise AssertionError("Invalid arguments to allgather")
        
# Get dense representation of local part of tensor
def to_dense(obj):
    assert isinstance(obj, (PETScMatrix, PETScVector))
    if isinstance(obj, PETScMatrix):
        ai, aj, av = obj.mat().getValuesCSR()
        return csr_matrix((av, aj, ai), shape=(obj.mat().getLocalSize()[0], obj.mat().getSize()[1])).todense().A
    elif isinstance(obj, PETScVector):
        return obj.get_local()
