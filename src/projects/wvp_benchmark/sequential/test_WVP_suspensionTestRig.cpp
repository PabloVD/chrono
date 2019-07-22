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
// WVP suspension test rig.
//
// Driver inputs for a suspension test rig include left/right post displacements
// and steering input (the latter being ignored if the tested suspension is not
// attached to a steering mechanism).  These driver inputs can be obtained from
// an interactive driver system (of type ChIrrGuiDriverSTR) or from a data file
// (using a driver system of type ChDataDriverSTR).
//
// If data collection is enabled, an output file named 'output.dat' will be
// generated in the directory specified by the variable out_dir. This ASCII file
// contains one line per output time, each with the following information:
//  [col  1]     time
//  [col  2]     left post input, a value in [-1,1]
//  [col  3]     right post input, a value in [-1,1]
//  [col  4]     steering input, a value in [-1,1]
//  [col  5]     actual left post dispalcement
//  [col  6]     actual right post displacement
//  [col  7- 9]  application point for left tire force
//  [col 10-12]  left tire force
//  [col 13-15]  left tire moment
//  [col 16-18]  application point for right tire force
//  [col 19-21]  right tire force
//  [col 22-24]  right tire moment
//
// Tire forces are expressed in the global frame, as applied to the center of
// the associated wheel.
//
// =============================================================================

#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/wheeled_vehicle/test_rig/ChSuspensionTestRig.h"
#include "chrono_vehicle/wheeled_vehicle/test_rig/ChIrrGuiDriverSTR.h"
#include "chrono_vehicle/wheeled_vehicle/test_rig/ChDataDriverSTR.h"
#include "chrono_vehicle/wheeled_vehicle/utils/ChWheeledVehicleIrrApp.h"

#include "chrono_models/vehicle/wvp/WVP_Vehicle.h"
#include "chrono_models/vehicle/wvp/WVP_RigidTire.h"
#include "chrono_models/vehicle/wvp/WVP_TMeasyTire.h"

#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
using namespace chrono::vehicle;
using namespace chrono::vehicle::wvp;

// =============================================================================
// USER SETTINGS
// =============================================================================

// Simulation step size
double step_size = 1e-3;

// Axle index
int axle_index = 0;
double post_limit = 0.15;

// Specification of test rig inputs:
//   'true':  use driver inputs from file
//   'false': use interactive Irrlicht driver
bool use_data_driver = false;

// File with driver inputs
std::string driver_file("hmmwv/suspensionTest/ST_inputs.dat");

// Output collection
bool collect_output = false;
std::string out_dir = "../WVP_TEST_RIG";
double out_step_size = 1.0 / 100;

// =============================================================================
int main(int argc, char* argv[]) {
    // Create and initialize a WVP vehicle.
    WVP_Vehicle vehicle(true);
    vehicle.Initialize(ChCoordsys<>(ChVector<>(0, 0, 0), ChQuaternion<>(1, 0, 0, 0)));

    // Use rigid wheels to actuate suspension.
    ////auto tire_L = std::make_shared<WVP_RigidTire>("Left", true);
    ////auto tire_R = std::make_shared<WVP_RigidTire>("Right", true);
    auto tire_L = std::make_shared<WVP_TMeasyTire>("Left");
    auto tire_R = std::make_shared<WVP_TMeasyTire>("Right");

    // Create and intialize the suspension test rig.
    ChSuspensionTestRig rig(vehicle, axle_index, post_limit, tire_L, tire_R);

    rig.SetInitialRideHeight(0.5);

    rig.SetSuspensionVisualizationType(VisualizationType::PRIMITIVES);
    rig.SetWheelVisualizationType(VisualizationType::PRIMITIVES);
    if (rig.HasSteering()) {
        rig.SetSteeringVisualizationType(VisualizationType::PRIMITIVES);
    }
    rig.SetTireVisualizationType(VisualizationType::PRIMITIVES);

    // Create the vehicle Irrlicht application.
    ChVehicleIrrApp app(&rig, NULL, L"WVP Suspension Test Rig");
    app.SetSkyBox();
    app.AddTypicalLights(irr::core::vector3df(30.f, -30.f, 100.f), irr::core::vector3df(30.f, 50.f, 100.f), 250, 130);
    app.SetChaseCamera(0.5 * (rig.GetWheelPos(LEFT) + rig.GetWheelPos(RIGHT)), 2.0, 1.0);
    app.SetTimestep(step_size);

    // Create and initialize the driver system.
    std::unique_ptr<ChDriverSTR> driver;
    if (use_data_driver) {
        // Driver with inputs from file
        auto data_driver = new ChDataDriverSTR(vehicle::GetDataFile(driver_file));
        driver = std::unique_ptr<ChDriverSTR>(data_driver);
    } else {
        // Interactive driver
        auto irr_driver = new ChIrrGuiDriverSTR(app);
        double steering_time = 1.0;      // time to go from 0 to max
        double displacement_time = 2.0;  // time to go from 0 to max applied post motion
        driver = std::unique_ptr<ChDriverSTR>(irr_driver);
    }
    rig.SetDriver(std::move(driver));

    // Initialize suspension test rig.
    rig.Initialize();

    app.AssetBindAll();
    app.AssetUpdateAll();

    // Initialize output
    if (!filesystem::create_directory(filesystem::path(out_dir))) {
        std::cout << "Error creating directory " << out_dir << std::endl;
        return 1;
    }
    std::string out_file = out_dir + "/output.dat";
    utils::CSV_writer out_csv(" ");

    // ---------------
    // Simulation loop
    // ---------------

    // Number of simulation steps between two data collection frames
    int out_steps = (int)std::ceil(out_step_size / step_size);

    // Initialize simulation frame counter
    int step_number = 0;

    while (app.GetDevice()->run()) {
        // Render scene
        app.BeginScene(true, true, irr::video::SColor(255, 140, 161, 192));
        app.DrawAll();
        app.EndScene();

        // Write output data
        if (collect_output && step_number % out_steps == 0) {
            // Current tire forces
            auto tire_force_L = rig.GetTireForce(VehicleSide::LEFT);
            auto tire_force_R = rig.GetTireForce(VehicleSide::RIGHT);
            out_csv << rig.GetDisplacementLeftInput() << rig.GetDisplacementRightInput() << rig.GetSteeringInput();
            out_csv << rig.GetActuatorDisp(VehicleSide::LEFT) << rig.GetActuatorDisp(VehicleSide::RIGHT);
            out_csv << tire_force_L.point << tire_force_L.force << tire_force_L.moment;
            out_csv << tire_force_R.point << tire_force_R.force << tire_force_R.moment;
            out_csv << std::endl;
        }

        // Advance simulation of the rig
        rig.Advance(step_size);

        // Update visualization app
        app.Synchronize(tire_L->GetTemplateName(), rig.GetSteeringInput(), 0, 0);
        app.Advance(step_size);

        // Increment frame number
        step_number++;
    }

    // Write output file
    if (collect_output) {
        out_csv.write_to_file(out_file);
    }

    return 0;
}
