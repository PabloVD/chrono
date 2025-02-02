// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Radu Serban, Justin Madsen
// =============================================================================
//
// GD250 wheel subsystem
//
// =============================================================================

#include <algorithm>

#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_models/vehicle/g-wagon/gd250_Wheel.h"
#include "chrono_thirdparty/filesystem/path.h"

namespace chrono {
namespace vehicle {
namespace gwagon {

const double GD250_Wheel::m_mass = 12.0;
const ChVector<> GD250_Wheel::m_inertia(0.240642, 0.410903, 0.240642);

const double GD250_Wheel::m_radius = 0.2032;
const double GD250_Wheel::m_width = 0.1524;

GD250_Wheel::GD250_Wheel(const std::string& name) : ChWheel(name) {
    m_vis_mesh_file = "g-wagon/gd250_rim.obj";
}

}  // namespace gwagon
}  // end namespace vehicle
}  // end namespace chrono
