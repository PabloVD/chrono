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
// Authors: Radu Serban
// =============================================================================
//
// M113 idler subsystem.
//
// =============================================================================

#include "chrono/assets/ChTriangleMeshShape.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono_vehicle/ChVehicleModelData.h"

#include "chrono_models/vehicle/m113a/M113a_Idler.h"

#include "chrono_thirdparty/filesystem/path.h"

namespace chrono {
namespace vehicle {
namespace m113 {

// -----------------------------------------------------------------------------
// Static variables
// -----------------------------------------------------------------------------
const double M113a_Idler::m_wheel_mass = 44.2;
const ChVector<> M113a_Idler::m_wheel_inertia(0.683, 1.06, 0.683);
const double M113a_Idler::m_wheel_radius = 0.219;
const double M113a_Idler::m_wheel_width = 0.18542;
const double M113a_Idler::m_wheel_gap = 0.033;

const double M113a_Idler::m_carrier_mass = 50;
const ChVector<> M113a_Idler::m_carrier_inertia(2, 2, 2);
const double M113a_Idler::m_carrier_radius = 0.02;

const double M113a_Idler::m_tensioner_l0 = 0.75;
const double M113a_Idler::m_tensioner_f = 5e4;//2e4;
const double M113a_Idler::m_tensioner_k = 1e7;
const double M113a_Idler::m_tensioner_c = 4e4;

const std::string M113a_IdlerLeft::m_meshFile = "M113/meshes/Idler_L.obj";
const std::string M113a_IdlerRight::m_meshFile = "M113/meshes/Idler_R.obj";

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
class M113a_TensionerForce : public ChLinkTSDA::ForceFunctor {
  public:
    M113a_TensionerForce(double k, double c, double f, double l0) : m_k(k), m_c(c), m_f(f), m_l0(l0) {}

    virtual double evaluate(double time,
                            double rest_length,
                            double length,
                            double vel,
                            const ChLinkTSDA& link) override {
        return m_f - m_k * (length - m_l0) - m_c * vel;
    }

  private:
    double m_l0;
    double m_k;
    double m_c;
    double m_f;
};

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
M113a_Idler::M113a_Idler(const std::string& name) : ChDoubleIdler(name) {
    m_tensionerForceCB =
        chrono_types::make_shared<M113a_TensionerForce>(m_tensioner_k, m_tensioner_c, m_tensioner_f, m_tensioner_l0);
}

void M113a_Idler::CreateContactMaterial(ChContactMethod contact_method) {
    MaterialInfo minfo;
    minfo.mu = 0.7f;
    minfo.cr = 0.1f;
    minfo.Y = 1e7f;
    m_material = minfo.CreateMaterial(contact_method);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void M113a_Idler::AddVisualizationAssets(VisualizationType vis) {
    ChDoubleIdler::AddVisualizationAssets(vis);

    if (vis == VisualizationType::MESH) {
        auto trimesh = chrono_types::make_shared<geometry::ChTriangleMeshConnected>();
        trimesh->LoadWavefrontMesh(GetMeshFile(), false, false);
        auto trimesh_shape = chrono_types::make_shared<ChTriangleMeshShape>();
        trimesh_shape->SetMesh(trimesh);
        trimesh_shape->SetName(filesystem::path(GetMeshFile()).stem());
        trimesh_shape->SetMutable(false);
        m_wheel->AddVisualShape(trimesh_shape);
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
const ChVector<> M113a_Idler::GetLocation(PointId which) {
    ChVector<> point;

    switch (which) {
        case WHEEL:
            point = ChVector<>(0, 0, 0);
            break;
        case CARRIER:
            point = ChVector<>(0, -0.1, 0);
            break;
        case CARRIER_CHASSIS:
            point = ChVector<>(0, -0.2, 0);
            break;
        case TSDA_CARRIER:
            point = ChVector<>(0, -0.2, 0);
            break;
        case TSDA_CHASSIS:
            point = ChVector<>(0.5, -0.2, 0);
            break;
        default:
            point = ChVector<>(0, 0, 0);
            break;
    }

    if (GetVehicleSide() == RIGHT)
        point.y() *= -1;

    return point;
}

}  // end namespace m113
}  // end namespace vehicle
}  // end namespace chrono
