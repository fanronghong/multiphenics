// Copyright (C) 2016-2019 by the multiphenics authors
//
// This file is part of multiphenics.
//
// multiphenics is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// multiphenics is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with multiphenics. If not, see <http://www.gnu.org/licenses/>.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace multiphenics_wrappers
{
  void fem(py::module& m)
  {
    // multiphenics::fem::BlockDofMap
    py::class_<multiphenics::fem::BlockDofMap, std::shared_ptr<multiphenics::fem::BlockDofMap>, dolfin::fem::GenericDofMap>
      (m, "BlockDofMap", "multiphenics BlockDofMap object")
      .def(py::init<std::vector<std::shared_ptr<const dolfin::fem::GenericDofMap>>,
                    std::vector<std::vector<std::shared_ptr<const dolfin::mesh::MeshFunction<bool>>>>,
                    const dolfin::mesh::Mesh&>())
      .def("dofmaps", &multiphenics::fem::BlockDofMap::dofmaps)
      .def("ownership_range", &multiphenics::fem::BlockDofMap::ownership_range)
      .def("global_dimension", &multiphenics::fem::BlockDofMap::global_dimension)
      .def("original_to_block", &multiphenics::fem::BlockDofMap::original_to_block)
      .def("block_to_original", &multiphenics::fem::BlockDofMap::block_to_original)
      .def("sub_index_map", &multiphenics::fem::BlockDofMap::sub_index_map);
      
    // multiphenics::fem::BlockForm1
    py::class_<multiphenics::fem::BlockForm1, std::shared_ptr<multiphenics::fem::BlockForm1>>
      (m, "BlockForm1", "multiphenics BlockForm1 object")
      .def(py::init<std::vector<std::shared_ptr<const dolfin::fem::Form>>,
                    std::vector<std::shared_ptr<const multiphenics::function::BlockFunctionSpace>>>())
      .def("mesh", &multiphenics::fem::BlockForm1::mesh)
      .def("block_size", &multiphenics::fem::BlockForm1::block_size);
                    
    // multiphenics::fem::BlockForm2
    py::class_<multiphenics::fem::BlockForm2, std::shared_ptr<multiphenics::fem::BlockForm2>>
      (m, "BlockForm2", "multiphenics BlockForm2 object")
      .def(py::init<std::vector<std::vector<std::shared_ptr<const dolfin::fem::Form>>>,
                    std::vector<std::shared_ptr<const multiphenics::function::BlockFunctionSpace>>>())
      .def("mesh", &multiphenics::fem::BlockForm2::mesh)
      .def("block_size", &multiphenics::fem::BlockForm2::block_size);
    
    // multiphenics::fem::block_assemble
    m.def("block_assemble",
      py::overload_cast<
        const multiphenics::fem::BlockForm1&
      >(&multiphenics::fem::block_assemble),
      py::arg("L"));
    m.def("block_assemble",
      py::overload_cast<
        dolfin::la::PETScVector&, const multiphenics::fem::BlockForm1&
      >(&multiphenics::fem::block_assemble),
      py::arg("b"), py::arg("L"));
    m.def("block_assemble",
      py::overload_cast<
        const multiphenics::fem::BlockForm2&
      >(&multiphenics::fem::block_assemble),
      py::arg("a"));
    m.def("block_assemble",
      py::overload_cast<
        dolfin::la::PETScMatrix&, const multiphenics::fem::BlockForm2&
      >(&multiphenics::fem::block_assemble),
      py::arg("A"), py::arg("a"));
              
    // multiphenics::fem::BlockDirichletBC
    py::class_<multiphenics::fem::BlockDirichletBC, std::shared_ptr<multiphenics::fem::BlockDirichletBC>, dolfin::common::Variable>
      (m, "BlockDirichletBC", "multiphenics BlockDirichletBC object")
      .def(py::init<std::vector<std::vector<std::shared_ptr<const DirichletBC>>>,
                    std::shared_ptr<const BlockFunctionSpace>>())
      .def("block_function_space", &multiphenics::fem::BlockDirichletBC::block_function_space)
      .def("get_boundary_values", [](const multiphenics::fem::BlockDirichletBC& instance)
           {
             multiphenics::fem::BlockDirichletBC::Map map;
             instance.get_boundary_values(map);
             return map;
           });
  }
}
