// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2022 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Author: Radu Serban
// =============================================================================
//
// Generator functions for Polaris on SPH terrain system
//
// =============================================================================

#pragma once

#include "chrono/physics/ChSystem.h"
#include "chrono_vehicle/wheeled_vehicle/vehicle/WheeledVehicle.h"
#include "chrono_fsi/ChSystemFsi.h"

enum class PolarisModel { ORIGINAL, MODIFIED };

void CreateTerrain(chrono::ChSystem& sys,
                   chrono::fsi::ChSystemFsi& sysFSI,
                   const std::string& terrain_dir,
                   bool terrain_mesh_vis,
                   bool terrain_mesh_contact);

void CreateMeshMarkers(const chrono::geometry::ChTriangleMeshConnected& mesh,
                       double delta,
                       std::vector<chrono::ChVector<>>& point_cloud);

std::shared_ptr<chrono::vehicle::WheeledVehicle> CreateVehicle(PolarisModel model,
                                                               chrono::ChSystem& sys,
                                                               const chrono::ChCoordsys<>& init_pos,
                                                               chrono::fsi::ChSystemFsi& sysFSI);
