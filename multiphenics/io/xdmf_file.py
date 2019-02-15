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

import os
from dolfin.io import XDMFFile as dolfin_XDMFFile
from multiphenics.mesh import MeshRestriction

class MeshRestrictionXDMFFile(object):
    def __init__(self, mpi_comm, filename, encoding):
        self.filename = filename
        self.encoding = encoding
    
    def read_mesh_restriction(self, mesh):
        # Create empty MeshRestriction
        content = MeshRestriction(mesh)
        # Read in MeshFunctions
        D = mesh.topology.dim
        for d in range(D + 1):
            mesh_function_d_filename = self.filename + "/mesh_function_" + str(d) + ".xdmf"
            xdmf_file = XDMFFile(mesh.mpi_comm(), mesh_function_d_filename, self.encoding)
            mesh_function_d = xdmf_file.read_mf_bool(mesh)
            assert mesh_function_d.dim == d
            content.append(mesh_function_d)
        # Return
        return content
    
    def write(self, content):
        # Create output folder
        try:
            os.makedirs(self.filename)
        except OSError:
            if not os.path.isdir(self.filename):
                raise
        # Write out MeshFunctions
        for (d, mesh_function_d) in enumerate(content):
            mesh_function_d_filename = self.filename + "/mesh_function_" + str(d) + ".xdmf"
            xdmf_file = XDMFFile(mesh_function_d.mesh().mpi_comm(), mesh_function_d_filename, self.encoding)
            xdmf_file.write(mesh_function_d)
            
def XDMFFile(mpi_comm, filename, encoding=dolfin_XDMFFile.Encoding.HDF5):
    if filename.endswith(".rtc.xdmf"):
        return MeshRestrictionXDMFFile(mpi_comm, filename, encoding)
    else:
        return dolfin_XDMFFile(mpi_comm, filename, encoding)
