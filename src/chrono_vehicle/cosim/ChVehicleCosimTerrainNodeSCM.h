// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2020 projectchrono.org
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
// Definition of the SCM deformable TERRAIN NODE.
//
// The global reference frame has Z up, X towards the front of the vehicle, and
// Y pointing to the left.
//
// =============================================================================

#ifndef CH_VEHCOSIM__TERRAINNODE_SCM_H
#define CH_VEHCOSIM__TERRAINNODE_SCM_H

#include "chrono/physics/ChSystem.h"
#include "chrono_vehicle/cosim/ChVehicleCosimTerrainNode.h"
#include "chrono_vehicle/terrain/SCMDeformableTerrain.h"

namespace chrono {

#ifdef CHRONO_IRRLICHT
namespace irrlicht {
class ChIrrApp;
}
#endif

namespace vehicle {

class CH_VEHICLE_API ChVehicleCosimTerrainNodeSCM : public ChVehicleCosimTerrainNode {
  public:
    /// Create a rigid terrain subsystem.
    ChVehicleCosimTerrainNodeSCM(bool render,     ///< use OpenGL rendering
                                 int num_threads  ///< number of OpenMP threads for SCM ray-casting
    );
    ~ChVehicleCosimTerrainNodeSCM();

    virtual ChSystem* GetSystem() override { return m_system; }

    /// Set the SCM material properties for terrain.
    void SetPropertiesSCM(
        double spacing,        ///< SCM grid spacing
        double Bekker_Kphi,    ///< Kphi, frictional modulus in Bekker model
        double Bekker_Kc,      ///< Kc, cohesive modulus in Bekker model
        double Bekker_n,       ///< n, exponent of sinkage in Bekker model (usually 0.6...1.8)
        double Mohr_cohesion,  ///< cohesion [Pa], for shear failure
        double Mohr_friction,  ///< Friction angle [degrees], for shear failure
        double Janosi_shear,   ///< shear parameter J [m], (usually a few mm or cm)
        double elastic_K,      ///< elastic stiffness K per unit area, [Pa/m] (must be larger than Kphi)
        double damping_R       ///< vertical damping R per unit area [Pa.s/m] (proportional to vertical speed)
    );

    /// Initialize SCM terrain from the specified checkpoint file (which must exist in the output directory).
    /// By default, a flat rectangular SCM terrain patch is used. 
    void SetInputFromCheckpoint(const std::string& filename);

    /// Write checkpoint to the specified file (which will be created in the output directory).
    virtual void WriteCheckpoint(const std::string& filename) override;

  private:
    ChSystem* m_system;               ///< containing system
    SCMDeformableTerrain* m_terrain;  ///< SCM terrain
#ifdef CHRONO_IRRLICHT
    irrlicht::ChIrrApp* m_irrapp;     ///< Irrlicht run-time visualizatino
#endif

    double m_spacing;        ///< SCM grid spacing
    double m_Bekker_Kphi;    ///< Kphi, frictional modulus in Bekker model
    double m_Bekker_Kc;      ///< Kc, cohesive modulus in Bekker model
    double m_Bekker_n;       ///< n, exponent of sinkage in Bekker model (usually 0.6...1.8)
    double m_Mohr_cohesion;  ///< Cohesion in, Pa, for shear failure
    double m_Mohr_friction;  ///< Friction angle (in degrees!), for shear failure
    double m_Janosi_shear;   ///< J , shear parameter, in meters, in Janosi-Hanamoto formula (usually few mm or cm)
    double m_elastic_K;      ///< elastic stiffness K per unit area [Pa/m]
    double m_damping_R;      ///< vetical damping R per unit area [Pa.s/m]

    bool m_use_checkpoint;              ///< if true, initialize height from checkpoint file
    std::string m_checkpoint_filename;  ///< name of input checkpoint file

    virtual bool SupportsFlexibleTire() const override { return false; }  //// TODO

    virtual void Construct() override;

    virtual void CreateMeshProxies() override;
    virtual void UpdateMeshProxies() override;
    virtual void GetForcesMeshProxies() override;
    virtual void PrintMeshProxiesUpdateData() override;
    virtual void PrintMeshProxiesContactData() override;

    virtual void CreateWheelProxy() override;
    virtual void UpdateWheelProxy() override;
    virtual void GetForceWheelProxy() override;
    virtual void PrintWheelProxyUpdateData() override;
    virtual void PrintWheelProxyContactData() override;

    virtual void OutputTerrainData(int frame) override;
    virtual void OnSynchronize(int step_number, double time) override;
    virtual void OnAdvance(double step_size) override;
};

}  // end namespace vehicle
}  // end namespace chrono

#endif
