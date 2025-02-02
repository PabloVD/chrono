// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Radu Serban
// =============================================================================
//
// Rigid terrain
//
// =============================================================================

#include <cmath>
#include <cstdlib>

#include "chrono/assets/ChBoxShape.h"
#include "chrono/assets/ChTexture.h"
#include "chrono/physics/ChMaterialSurfaceNSC.h"
#include "chrono/physics/ChMaterialSurfaceSMC.h"

#include "RigidTerrainStep.h"

namespace chrono {
namespace vehicle {

// -----------------------------------------------------------------------------
// Constructor.
// -----------------------------------------------------------------------------
RigidTerrainStep::RigidTerrainStep(ChSystem* system) : m_friction(0.8f) {
    // Create the ground body and add it to the system.
    m_ground = std::shared_ptr<ChBody>(system->NewBody());
    m_ground->SetIdentifier(-1);
    m_ground->SetName("ground");
    m_ground->SetPos(ChVector<>(0, 0, 0));
    m_ground->SetBodyFixed(true);
    m_ground->SetCollide(true);
    system->AddBody(m_ground);
}

// -----------------------------------------------------------------------------
// Set the texture and texture scaling
// -----------------------------------------------------------------------------
void RigidTerrainStep::SetTexture(const std::string tex_file, float tex_scale_x, float tex_scale_y) {
    m_ground->GetVisualShape(0)->SetTexture(tex_file, tex_scale_x, tex_scale_y);
}

// -----------------------------------------------------------------------------
// Initialize the terrain as a rigid box
// -----------------------------------------------------------------------------
void RigidTerrainStep::Initialize(std::shared_ptr<ChMaterialSurface> mat,
                                  double height1,
                                  double height2,
                                  double sizeX,
                                  double sizeY,
                                  bool tiled,
                                  double max_tile_size) {
    m_friction = mat->GetSfriction();

    double depth = 10 + std::abs(height2 - height1);
    ChVector<> size(sizeX, sizeY, depth);

    m_ground->GetCollisionModel()->ClearModel();
    CreateBox(mat, ChVector<>(-sizeX / 2, 0, height1 - depth / 2), size, tiled, max_tile_size);
    CreateBox(mat, ChVector<>(sizeX / 2, 0, height2 - depth / 2), size, tiled, max_tile_size);
    m_ground->GetCollisionModel()->BuildModel();

    m_height1 = height1;
    m_height2 = height2;
}

void RigidTerrainStep::CreateBox(std::shared_ptr<ChMaterialSurface> mat,
                                 ChVector<> center,
                                 ChVector<> size,
                                 bool tiled,
                                 double max_tile_size) {
    if (tiled) {
        //// TODO
    } else {
        m_ground->GetCollisionModel()->AddBox(mat, size.x(), size.y(), size.z(), center);
    }

    auto box = chrono_types::make_shared<ChBoxShape>(size.x(), size.y(), size.z());
    m_ground->AddVisualShape(box, ChFrame<>(center));
}

// -----------------------------------------------------------------------------
// Return the terrain height at the specified location
// -----------------------------------------------------------------------------
double RigidTerrainStep::GetHeight(const ChVector<>& loc) const {
    return (loc.x() < 0) ? m_height1 : m_height2;
}

// -----------------------------------------------------------------------------
// Return the terrain normal at the specified location
// -----------------------------------------------------------------------------
ChVector<> RigidTerrainStep::GetNormal(const ChVector<>& loc) const {
    return ChVector<>(0, 0, 1);
}

}  // end namespace vehicle
}  // end namespace chrono
