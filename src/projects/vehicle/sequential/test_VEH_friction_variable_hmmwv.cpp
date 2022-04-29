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
// Test tire models on terrain with varying coefficient of friction.
//
// The vehicle reference frame has Z up, X towards the front of the vehicle, and
// Y pointing to the left.
//
// =============================================================================

#include "chrono/physics/ChSystemNSC.h"
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/terrain/RigidTerrain.h"
#include "chrono_vehicle/wheeled_vehicle/utils/ChWheeledVehicleVisualSystemIrrlicht.h"

#include "chrono_models/vehicle/hmmwv/HMMWV.h"

#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
using namespace chrono::vehicle;
using namespace chrono::vehicle::hmmwv;

// =============================================================================

// Type of tire model (RIGID, TMEASY, FIALA, PAC89, or PACEJKA)
TireModelType tire_model = TireModelType::TMEASY;

// Simulation step size, end time
double step_size = 2e-3;
double tire_step_size = 1e-3;

// Output
bool output = true;
double out_fps = 50;
const std::string out_dir = "../HMMWV_FRICTION_VARIABLE";

// =============================================================================

class MyFrictionFunctor : public ChTerrain::FrictionFunctor {
  public:
    MyFrictionFunctor() : m_friction_left(0.2f), m_friction_right(0.9f) {}
    virtual float operator()(const ChVector<>& loc) override { return loc.y() > 0 ? m_friction_left : m_friction_right; }
    float m_friction_left;
    float m_friction_right;
};

// =============================================================================

int main(int argc, char* argv[]) {
    // Chrono system
    ////ChSystemNSC sys;
    ChSystemSMC sys;
    sys.Set_G_acc(ChVector<>(0, 0, -9.81));
    sys.SetSolverMaxIterations(150);
    sys.SetMaxPenetrationRecoverySpeed(4.0);
    sys.SetSolverType(ChSolver::Type::BARZILAIBORWEIN);

    // Create and initialize the vehicle
    HMMWV_Full hmmwv(&sys);
    hmmwv.SetInitPosition(ChCoordsys<>(ChVector<>(-90, 0, 0.7), QUNIT));
    hmmwv.SetPowertrainType(PowertrainModelType::SHAFTS);
    hmmwv.SetDriveType(DrivelineTypeWV::RWD);
    hmmwv.SetTireType(tire_model);
    hmmwv.SetTireStepSize(tire_step_size);
    hmmwv.Initialize();
    hmmwv.SetChassisVisualizationType(VisualizationType::NONE);
    hmmwv.SetSuspensionVisualizationType(VisualizationType::PRIMITIVES);
    hmmwv.SetSteeringVisualizationType(VisualizationType::PRIMITIVES);
    hmmwv.SetWheelVisualizationType(VisualizationType::NONE);
    hmmwv.SetTireVisualizationType(VisualizationType::PRIMITIVES);

    // Create the terrain
    RigidTerrain terrain(&sys);

    MaterialInfo minfo;
    minfo.mu = 0.9f;
    minfo.cr = 0.01f;
    minfo.Y = 2e7f;
    auto mat = minfo.CreateMaterial(sys.GetContactMethod());

    auto patch = terrain.AddPatch(mat, ChVector<>(0, 0, 0), ChVector<>(0, 0, 1), 200, 10);
    patch->SetColor(ChColor(0.8f, 0.8f, 1.0f));
    patch->SetTexture(vehicle::GetDataFile("terrain/textures/tile4.jpg"), 200, 10);

    MyFrictionFunctor ffun;
    terrain.RegisterFrictionFunctor(&ffun);
    terrain.UseLocationDependentFriction(true);

    terrain.Initialize();

    std::string modelname;
    switch (tire_model) {
        case TireModelType::TMEASY:
            modelname = "tmeasy";
            break;
        case TireModelType::FIALA:
            modelname = "fiala";
            break;
        case TireModelType::PAC89:
            modelname = "pac89";
            break;
        case TireModelType::PACEJKA:
            modelname = "pacejka";
            break;
        case TireModelType::RIGID:
            modelname = "rigid";
            break;
    }

    // Create the vehicle Irrlicht interface
    auto vis = chrono_types::make_shared<ChWheeledVehicleVisualSystemIrrlicht>();
    vis->SetWindowTitle("Terrain friction test");
    vis->SetChaseCamera(ChVector<>(0.0, 0.0, .75), 6.0, 0.5);
    vis->Initialize();
    vis->AddTypicalLights();
    vis->AddSkyBox();
    vis->AddLogo();
    hmmwv.GetVehicle().SetVisualSystem(vis);

    // Initialize output
    if (output) {
        if (!filesystem::create_directory(filesystem::path(out_dir))) {
            std::cout << "Error creating directory " << out_dir << std::endl;
            return 1;
        }
    }

    utils::CSV_writer csv("\t");
    csv.stream().setf(std::ios::scientific | std::ios::showpos);
    csv.stream().precision(6);

    // ---------------
    // Simulation loop
    // ---------------

    double out_step_size = 1 / out_fps;
    int out_steps = (int)std::ceil(out_step_size / step_size);
    int sim_frame = 0;

    while (vis->Run()) {
        double time = sys.GetChTime();

        auto veh_pos = hmmwv.GetVehicle().GetPos();

        // Extract output
        if (output && sim_frame % out_steps == 0) {
            double veh_speed = hmmwv.GetVehicle().GetSpeed();
            hmmwv.GetVehicle().GetAxle(0)->GetWheel(LEFT);
            double fl_omega = hmmwv.GetVehicle().GetSpindleOmega(0, LEFT);
            double fr_omega = hmmwv.GetVehicle().GetSpindleOmega(0, RIGHT);
            double rl_omega = hmmwv.GetVehicle().GetSpindleOmega(1, LEFT);
            double rr_omega = hmmwv.GetVehicle().GetSpindleOmega(1, RIGHT);

            csv << time << veh_pos.x() << veh_speed;
            csv << fl_omega << fr_omega << rl_omega << rr_omega << std::endl;
        }

        // Stop before end of terrain patch
        if (veh_pos.x() > 80 || std::abs(veh_pos.y()) > 4) {
            std::cout << "Longitudinal travel distance = " << veh_pos.x() + 90 << std::endl;
            break;
        }

        // Render scene
        vis->BeginScene();
        vis->DrawAll();
        vis->EndScene();

        // Driver inputs
        ChDriver::Inputs driver_inputs;
        driver_inputs.m_steering = 0;
        driver_inputs.m_braking = 0;
        driver_inputs.m_throttle = 0;
        if (time > 1 && time < 2)
            driver_inputs.m_throttle = 0.75 * (time - 1);
        else if (time > 2)
            driver_inputs.m_throttle = 0.75;

        // Update modules (process inputs from other modules)
        hmmwv.Synchronize(time, driver_inputs, terrain);
        terrain.Synchronize(time);
        vis->Synchronize(modelname, driver_inputs);

        // Advance simulation for one timestep for all modules.
        hmmwv.Advance(step_size);
        terrain.Advance(step_size);
        vis->Advance(step_size);

        // Advance state of entire system
        sys.DoStepDynamics(step_size);

        // Increment simulation frame number
        sim_frame++;
    }

    if (output) {
        csv.write_to_file(out_dir + "/" + modelname + ".out");
    }

    return 0;
}
