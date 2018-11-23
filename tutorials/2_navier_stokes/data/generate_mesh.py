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

from dolfin import *
from mshr import *

# Geometrical parameters
pre_step_length = 4.
after_step_length = 14.
pre_step_height = 3.
after_step_height = 5.

# Create mesh
domain = (
    Rectangle(Point(0., 0.), Point(pre_step_length + after_step_length, after_step_height)) -
    Rectangle(Point(0., 0.), Point(pre_step_length, after_step_height - pre_step_height))
)
mesh = generate_mesh(domain, 62)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, mesh.topology.dim, 0)

# Create boundaries
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]) < DOLFIN_EPS

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (
            (x[0] <= pre_step_length and abs(x[1] - after_step_height + pre_step_height) < DOLFIN_EPS) or
            (x[1] <= after_step_height - pre_step_height and abs(x[0] - pre_step_length) < DOLFIN_EPS) or
            (x[0] >= pre_step_length and abs(x[1]) < DOLFIN_EPS)
        )
        
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - after_step_height) < DOLFIN_EPS
    
boundaries = MeshFunction("size_t", mesh, mesh.topology.dim - 1, 0)
inlet = Inlet()
inlet_ID = 1
inlet.mark(boundaries, inlet_ID)
bottom = Bottom()
bottom_ID = 2
bottom.mark(boundaries, bottom_ID)
top = Top()
top_ID = 2
top.mark(boundaries, top_ID)

# Save
XDMFFile("backward_facing_step.xdmf").write(mesh)
XDMFFile("backward_facing_step_physical_region.xdmf").write(subdomains)
XDMFFile("backward_facing_step_facet_region.xdmf").write(boundaries)
