# Copyright (C) 2015-2016 by the RBniCS authors
# Copyright (C) 2016 by the block_ext authors
#
# This file is part of the RBniCS interface to block_ext.
#
# RBniCS and block_ext are free software: you can redistribute them and/or modify
# them under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS and block_ext are distributed in the hope that they will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS and block_ext. If not, see <http://www.gnu.org/licenses/>.
#

from block_ext import BlockDirichletBC, block_solve
from RBniCS.backends.abstract import LinearSolver as AbstractLinearSolver
from block_ext.RBniCS.backends.block_ext.matrix import Matrix
from block_ext.RBniCS.backends.block_ext.vector import Vector
from block_ext.RBniCS.backends.block_ext.function import Function
from RBniCS.utils.decorators import BackendFor, Extends, list_of, override

@Extends(AbstractLinearSolver)
@BackendFor("block_ext", inputs=(Matrix.Type(), Function.Type(), Vector.Type(), (list_of(BlockDirichletBC), None)))
class LinearSolver(AbstractLinearSolver):
    @override
    def __init__(self, lhs, solution, rhs, bcs=None): # TODO deve mettere il block discard dofs?
        assert False # TODO considerare le BCs
        
    @override
    def set_parameters(self, parameters):
        assert len(parameters) == 0, "block_ext linear solver does not accept parameters yet"
        
    @override
    def solve(self):
        block_solve(self.lhs, self.solution.block_vector(), self.rhs)
        return self.solution
        