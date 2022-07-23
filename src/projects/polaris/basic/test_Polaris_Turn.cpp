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
// Polaris turning radius test
// 
// =============================================================================

#include "chrono/core/ChRealtimeStep.h"
#include "chrono/utils/ChFilters.h"
#include "chrono/solver/ChIterativeSolverLS.h"

#include "chrono_vehicle/ChConfigVehicle.h"
#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/terrain/RigidTerrain.h"
#include "chrono_vehicle/driver/ChPathFollowerDriver.h"
#include "chrono_vehicle/utils/ChUtilsJSON.h"
#include "chrono_vehicle/utils/ChVehiclePath.h"

#include "chrono_vehicle/wheeled_vehicle/vehicle/WheeledVehicle.h"
#include "chrono_vehicle/wheeled_vehicle/vehicle/WheeledTrailer.h"
#include "chrono_vehicle/wheeled_vehicle/utils/ChWheeledVehicleVisualSystemIrrlicht.h"

#include "chrono_thirdparty/filesystem/path.h"

#ifdef CHRONO_POSTPROCESS
    #include "chrono_postprocess/ChGnuPlot.h"
#endif

using namespace chrono;
using namespace chrono::vehicle;

// =============================================================================

enum class MRZR_MODEL { ORIGINAL, MODIFIED };

MRZR_MODEL model = MRZR_MODEL::MODIFIED;

// =============================================================================

// Initial vehicle location and orientation
ChVector<> initLoc(0, 1, 10);
double initYaw = 0;
////double initYaw = CH_C_PI / 4;  // 45 deg towards vehicle's left


// =============================================================================

int main(int argc, char* argv[]) {
    std::string model_dir = (model == MRZR_MODEL::ORIGINAL) ? "mrzr/JSON_orig/" : "mrzr/JSON_new/";

    std::string vehicle_json = model_dir + "vehicle/MRZR.json";

    ////std::string powertrain_json = model_dir + "powertrain/MRZR_SimplePowertrain.json";
    std::string powertrain_json = model_dir + "powertrain/MRZR_SimpleMapPowertrain.json";

    ////std::string tire_json = model_dir + "tire/MRZR_RigidTire.json";
    ////std::string tire_json = model_dir + "tire/MRZR_TMeasyTire.json";
    std::string tire_json = model_dir + "tire/MRZR_Pac02Tire.json";

    // Create the vehicle system
    auto contact_method = ChContactMethod::SMC;
    WheeledVehicle vehicle(vehicle::GetDataFile(vehicle_json), contact_method);
    vehicle.Initialize(ChCoordsys<>(ChVector<>(0, 0, 0.2), QUNIT));
    vehicle.GetChassis()->SetFixed(false);
    vehicle.SetChassisVisualizationType(VisualizationType::MESH);
    vehicle.SetSuspensionVisualizationType(VisualizationType::PRIMITIVES);
    vehicle.SetSteeringVisualizationType(VisualizationType::PRIMITIVES);
    vehicle.SetWheelVisualizationType(VisualizationType::MESH);

    // Create and initialize the powertrain system
    auto powertrain = ReadPowertrainJSON(vehicle::GetDataFile(powertrain_json));
    vehicle.InitializePowertrain(powertrain);

    // Create and initialize the tires
    for (auto& axle : vehicle.GetAxles()) {
        for (auto& wheel : axle->GetWheels()) {
            auto tire = ReadTireJSON(vehicle::GetDataFile(tire_json));
            vehicle.InitializeTire(tire, wheel, VisualizationType::MESH);
        }
    }

    // Create the terrain
    RigidTerrain terrain(vehicle.GetSystem());
    auto patch_mat = chrono_types::make_shared<ChMaterialSurfaceSMC>();
    patch_mat->SetFriction(0.9f);
    patch_mat->SetRestitution(0.01f);
    patch_mat->SetYoungModulus(2e7f);
    patch_mat->SetPoissonRatio(0.3f);
    auto patch = terrain.AddPatch(patch_mat, CSYSNORM, 200, 200);
    patch->SetColor(ChColor(0.8f, 0.8f, 0.5f));
    patch->SetTexture(vehicle::GetDataFile("terrain/textures/tile4.jpg"), 200, 200);
    terrain.Initialize();

    // Path follower driver
    auto path = CirclePath(ChVector<>(0,0,0.5), 6, 20, true, 5);

    ChPathFollowerDriver driver(vehicle, path, "my_path", 3);
    driver.GetSteeringController().SetLookAheadDistance(2.0);
    driver.GetSteeringController().SetGains(0.5, 0, 0);
    driver.GetSpeedController().SetGains(0.8, 0, 0);
    driver.Initialize();

    // Create the vehicle Irrlicht interface
    auto vis = chrono_types::make_shared<ChWheeledVehicleVisualSystemIrrlicht>();
    vis->SetWindowTitle("Polaris acceleration test");
    vis->SetChaseCamera(ChVector<>(0.0, 0.0, 1.75), 5.0, 0.5);
    vis->Initialize();
    vis->AddSkyBox();
    vis->AddLogo();
    vis->AddTypicalLights();
    vis->AttachVehicle(&vehicle);

    // Visualization of controller points (sentinel & target)
    irr::scene::IMeshSceneNode* ballS = vis->GetSceneManager()->addSphereSceneNode(0.1f);
    irr::scene::IMeshSceneNode* ballT = vis->GetSceneManager()->addSphereSceneNode(0.1f);
    ballS->getMaterial(0).EmissiveColor = irr::video::SColor(0, 255, 0, 0);
    ballT->getMaterial(0).EmissiveColor = irr::video::SColor(0, 0, 255, 0);

    // ---------------
    // Simulation loop
    // ---------------
    double step_size = 1e-3;
    ChRealtimeStepTimer realtime_timer;
    utils::ChRunningAverage RTF_filter(50);

    while (vis->Run()) {
        double time = vehicle.GetChTime();

        const ChVector<>& pS = driver.GetSteeringController().GetSentinelLocation();
        const ChVector<>& pT = driver.GetSteeringController().GetTargetLocation();
        ballS->setPosition(irr::core::vector3df((irr::f32)pS.x(), (irr::f32)pS.y(), (irr::f32)pS.z()));
        ballT->setPosition(irr::core::vector3df((irr::f32)pT.x(), (irr::f32)pT.y(), (irr::f32)pT.z()));

        vis->BeginScene();
        vis->Render();
        vis->EndScene();

        // Driver inputs
        DriverInputs driver_inputs = driver.GetInputs();

        // Update modules (process inputs from other modules)
        driver.Synchronize(time);
        terrain.Synchronize(time);
        vehicle.Synchronize(time, driver_inputs, terrain);
        vis->Synchronize("Acceleration test", driver_inputs);

        // Advance simulation for one timestep for all modules
        driver.Advance(step_size);
        terrain.Advance(step_size);
        vehicle.Advance(step_size);
        vis->Advance(step_size);

        // Spin in place for real time to catch up
        realtime_timer.Spin(step_size);
        ////std::cout << RTF_filter.Add(realtime_timer.RTF) << std::endl;
    }

    return 0;
}
