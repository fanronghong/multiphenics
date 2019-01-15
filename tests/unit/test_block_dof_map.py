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

import pytest
import matplotlib.pyplot as plt
from ufl import MixedElement, TensorElement, VectorElement
from dolfin import near, SubDomain, UnitSquareMesh
from dolfin_utils.test import fixture as module_fixture
from multiphenics import BlockElement, BlockFunctionSpace
from test_utils import get_elements_1, get_elements_2

# Mesh
@module_fixture
def mesh():
    return UnitSquareMesh(1, 1)
    
# Restrictions (less cases than those in test_utils.py)
class Subdomain(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] - x[1] <= 0.
subdomain = Subdomain()

class Interface(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0] - x[1], 0.)
interface = Interface()
        
def get_restrictions_1():
    return (
        None,
        subdomain,
        interface
    )
    
def get_restrictions_2():
    return (
        (None, None),
        (None, subdomain),
        (None, interface),
        (subdomain, None),
        (subdomain, subdomain),
        (subdomain, interface),
        (interface, None),
        (interface, subdomain),
        (interface, interface)
    )
    
# Expected results
expected = {
    (
        ("Lagrange", 1, ()), # scalar CG 1
    ): {
        None: {
            (0., 0.): 1,
            (0., 1.): 1,
            (1., 0.): 1,
            (1., 1.): 1,
        },
        subdomain: {
            (0., 0.): 1,
            (0., 1.): 1,
            (1., 1.): 1,
        },
        interface: {
            (0., 0.): 1,
            (1., 1.): 1,
        },
    },
    (
        ("Lagrange", 2, ()), # scalar CG 2
    ): {
        None: {
            (0., 0.): 1,
            (0., 0.5): 1,
            (0., 1.): 1,
            (0.5, 0.): 1,
            (0.5, 0.5): 1,
            (0.5, 1.): 1,
            (1., 0.): 1,
            (1., 0.5): 1,
            (1., 1.): 1,
        },
        subdomain: {
            (0., 0.): 1,
            (0., 0.5): 1,
            (0., 1.): 1,
            (0.5, 0.5): 1,
            (0.5, 1.): 1,
            (1., 1.): 1,
        },
        interface: {
            (0., 0.): 1,
            (0.5, 0.5): 1,
            (1., 1.): 1,
        },
    },
    (
        ("Lagrange", 1, (2, )), # vector CG 1
    ): {
        None: {
            (0., 0.): 2,
            (0., 1.): 2,
            (1., 0.): 2,
            (1., 1.): 2,
        },
        subdomain: {
            (0., 0.): 2,
            (0., 1.): 2,
            (1., 1.): 2,
        },
        interface: {
            (0., 0.): 2,
            (1., 1.): 2,
        },
    },
    (
        ("Lagrange", 2, (2, )), # vector CG 2
    ): {
        # TODO
    },
    (
        ("Lagrange", 1, (2, 2)), # tensor CG 1
    ): {
        # TODO
    },
    (
        ("Lagrange", 2, (2, 2)), # tensor CG 2
    ): {
        # TODO
    },
    (
        ("Lagrange", 2, (2, )), ("Lagrange", 1, ()) # Stokes CG 2/CG 1
    ): {
        # TODO
    },
    (
        ("Lagrange", 3, (2, )), ("Lagrange", 2, ()) # Stokes CG 3/CG 2
    ): {
        # TODO
    },
    (
        ("Discontinuous Lagrange", 0, ()), # scalar DG 0
    ): {
        # TODO
    },
    (
        ("Discontinuous Lagrange", 1, ()), # scalar DG 1
    ): {
        # TODO
    },
    (
        ("Discontinuous Lagrange", 2, ()), # scalar DG 2
    ): {
        # TODO
    },
    (
        ("Real", 0, ()), # scalar Real 0
    ): {
        # TODO
    },
    (
        ("Real", 0, (2, )), # vector Real 0
    ): {
        # TODO
    },
    (
        ("Lagrange", 1, ()), ("Lagrange", 1, ()) # scalar CG 1 enriched with scalar Real 0
    ): {
        # TODO
    },
    (
        ("Lagrange", 2, ()), ("Lagrange", 2, ()) # scalar CG 2 enriched with scalar Real 0
    ): {
        # TODO
    },
}

# Auxiliary function for dofs counting
def count_dofs_at_coordinates(block_V):
    """
    This function returns a dict mapping each coordinate (as a tuple)
    to the number of DOFs associated to that coordinate.
    """
    xy = block_V.tabulate_dof_coordinates().round(decimals=3)
    xy_list = [tuple(coord) for coord in xy]
    xy_unique = list(set(xy_list))
    count = dict()
    for coord in xy_unique:
        count[coord] = xy_list.count(coord)
    return count
    
# Auxiliary function for plot
def plot(count):
    """
    This function shows the coordinates of each DOF, together
    with the corresponding cardinality as a label.
    """
    fig = plt.figure()
    sub = fig.add_subplot(111)
    x = [coord[0] for coord in count.keys()]
    y = [coord[1] for coord in count.keys()]
    sub.plot(x, y, 'bo')
    for (coord, count_coord) in count.items():
        sub.annotate(count_coord, xy=coord)
    plt.show()
    
# Auxiliary functions for asserts
def _ufl_element_conversion(element, result=None):
    if result is None:
        should_return = True
        result = list()
    else:
        should_return = False
    if isinstance(element, MixedElement) and not isinstance(element, (TensorElement, VectorElement)):
        for sub_element in element.sub_elements():
            _ufl_element_conversion(sub_element, result)
    else:
        result.append((element.family(), element.degree(), element.value_shape()))
    if should_return:
        return tuple(result)
        
def ufl_element_conversion(element1, element2=None):
    converted_element1 = _ufl_element_conversion(element1)
    if element2 is not None:
        converted_element2 = _ufl_element_conversion(element2)
        return (converted_element1, converted_element2)
    else:
        return converted_element1
        
def assert_count_1(element, restriction, count):
    converted_element = ufl_element_conversion(element)
    assert converted_element in expected
    assert restriction in expected[converted_element]
    assert count == expected[converted_element][restriction]
    
def assert_count_2(element, restriction, count):
    restriction = tuple(restriction)
    converted_element = ufl_element_conversion(*element)
    assert converted_element[0] in expected
    assert converted_element[1] in expected
    assert restriction[0] in expected[converted_element[0]]
    assert restriction[1] in expected[converted_element[1]]
    expected_0 = expected[converted_element[0]][restriction[0]]
    expected_1 = expected[converted_element[1]][restriction[1]]
    expected_01 = { coord: expected_0.get(coord, 0) + expected_1.get(coord, 0) for coord in set(expected_0.keys()) | set(expected_1.keys()) }
    assert count == expected_01

# Single block, with restriction, from block element
@pytest.mark.parametrize("restriction", get_restrictions_1())
@pytest.mark.parametrize("Element", get_elements_1())
def test_single_block_with_restriction_from_block_element(mesh, restriction, Element):
    V_element = Element(mesh)
    block_V_element = BlockElement(V_element)
    block_V = BlockFunctionSpace(mesh, block_V_element, restrict=[restriction])
    count = count_dofs_at_coordinates(block_V)
    plot(count)
    assert_count_1(V_element, restriction, count)
        
# Two blocks, with restrictions, from block element
@pytest.mark.parametrize("restriction", get_restrictions_2())
@pytest.mark.parametrize("Elements", get_elements_2())
def test_two_blocks_with_restriction_from_block_element(mesh, restriction, Elements):
    (Element1, Element2) = Elements
    V1_element = Element1(mesh)
    V2_element = Element2(mesh)
    block_V_element = BlockElement(V1_element, V2_element)
    block_V = BlockFunctionSpace(mesh, block_V_element, restrict=restriction)
    count = count_dofs_at_coordinates(block_V)
    plot(count)
    assert_count_2((V1_element, V2_element), restriction, count)
