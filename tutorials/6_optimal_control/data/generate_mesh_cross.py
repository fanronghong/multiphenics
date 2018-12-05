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

from dolfin import *
from mshr import *
from multiphenics import *

"""
This file generates the mesh which is used in the following examples:
    3c_poisson
The test case is from section 3 of
M. Stoll and A. Wathen, All-at-once solution of time-dependent PDE-constrained optimization problems, Oxford Centre for Collaborative Applied Mathematics, Technical Report 10/47, 2010.
"""

# Geometrical parameters
mu1 = 0.4
mu2 = 0.6

# Create mesh
domain = Rectangle(Point(0., 0.), Point(1., 1.))
square_1 = Rectangle(Point(0., 0.), Point(mu1, mu1))
square_2 = Rectangle(Point(mu2, 0.), Point(1., mu1))
square_3 = Rectangle(Point(0., mu2), Point(mu1, 1.))
square_4 = Rectangle(Point(mu2, mu2), Point(1., 1.))
cross = domain - square_1 - square_2 - square_3 - square_4
domain.set_subdomain(1, cross)
mesh = generate_mesh(domain, 32)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())

# Create boundaries
class OnBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
        
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
on_boundary = OnBoundary()
on_boundary.mark(boundaries, 1)

# Save
File("cross.xml") << mesh
File("cross_physical_region.xml") << subdomains
File("cross_facet_region.xml") << boundaries
XDMFFile("cross.xdmf").write(mesh)
XDMFFile("cross_physical_region.xdmf").write(subdomains)
XDMFFile("cross_facet_region.xdmf").write(boundaries)
