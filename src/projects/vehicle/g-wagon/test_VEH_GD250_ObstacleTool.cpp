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
// Authors: Rainer Gericke
// =============================================================================
//
// Demonstration program for GD250 vehicle on rigid trapezoidal obstacle
//
// =============================================================================

#include <iomanip>

#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono/utils/ChFilters.h"
#include "chrono/core/ChTimer.h"
#include "chrono/solver/ChSolverBB.h"

#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/driver/ChInteractiveDriverIRR.h"
#include "chrono_vehicle/driver/ChPathFollowerDriver.h"

#include "chrono_vehicle/terrain/ObsModTerrain.h"
#include "chrono_vehicle/output/ChVehicleOutputASCII.h"

#include "chrono_vehicle/wheeled_vehicle/vehicle/WheeledVehicle.h"
#include "chrono_models/vehicle/g-wagon/gd250.h"

#include "chrono_vehicle/wheeled_vehicle/ChWheeledVehicleVisualSystemIrrlicht.h"

#ifdef CHRONO_PARDISO_MKL
    #include "chrono_pardisomkl/ChSolverPardisoMKL.h"
#endif

#include "chrono_thirdparty/filesystem/path.h"

using namespace chrono;
using namespace chrono::utils;
using namespace chrono::vehicle;
using namespace chrono::vehicle::gwagon;

using std::cout;
using std::endl;

// =============================================================================
// USER SETTINGS
// =============================================================================
// Initial vehicle position
ChVector<> initLoc(-1, 0, 0.4);

// Initial vehicle orientation
ChQuaternion<> initRot(1, 0, 0, 0);
// ChQuaternion<> initRot(0.866025, 0, 0, 0.5);
// ChQuaternion<> initRot(0.7071068, 0, 0, 0.7071068);
// ChQuaternion<> initRot(0.25882, 0, 0, 0.965926);
// ChQuaternion<> initRot(0, 0, 0, 1);

// Rigid terrain dimensions
double terrainHeight = 0;
double terrainLength = 100.0;  // size in X direction
double terrainWidth = 100.0;   // size in Y direction

// Simulation step size
double step_size = 1e-3;

// Use HHT + MKL
bool use_mkl = false;

// Time interval between two render frames
double render_step_size = 1.0 / 120;  // FPS = 120

// Point on chassis tracked by the camera
ChVector<> trackPoint(0.0, 2.0, 0.0);

// Type of tire model (RIGID, TMEASY)
TireModelType tire_model = TireModelType::TMEASY;

// Driver input files
std::string path_file("paths/straightOrigin.txt");
std::string steering_controller_file("hmmwv/SteeringController.json");
std::string speed_controller_file("hmmwv/SpeedController.json");

// Output directories
const std::string out_top_dir = GetChronoOutputPath() + "GD250";
const std::string out_dir = out_top_dir + "/OBSTACLE";
const std::string pov_dir = out_dir + "/POVRAY";
const std::string img_dir = out_dir + "/IMG";

// Output
bool povray_output = false;
bool img_output = false;
bool dbg_output = false;

// =============================================================================
const double mph_to_ms = 0.44704;

// =============================================================================
int main(int argc, char* argv[]) {
    GetLog() << "Copyright (c) 2021 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";

    const double inchToMeters = 0.0254;
    const double MetersToInch = 1.0 / 0.0254;
    const double NewtonToLbf = 0.2248089431;

    double target_speed = 1.5;
    double xpos_max = 100.0;

    std::vector<double> angles;   // winkel in rad
    std::vector<double> widths;   // länge in in
    std::vector<double> heights;  // höhen in in

    heights.push_back(0.1 * MetersToInch);
    heights.push_back(0.25 * MetersToInch);
    heights.push_back(0.5 * MetersToInch);
    heights.push_back(0.75 * MetersToInch);
    heights.push_back(1.0 * MetersToInch);

    widths.push_back(1.0 * MetersToInch);
    widths.push_back(2.0 * MetersToInch);
    widths.push_back(3.0 * MetersToInch);
    widths.push_back(4.0 * MetersToInch);
    widths.push_back(5.0 * MetersToInch);

    angles.push_back((180.0 - 45.0) * CH_C_DEG_TO_RAD);
    angles.push_back((180.0 - 30.0) * CH_C_DEG_TO_RAD);
    angles.push_back((180.0 - 15.0) * CH_C_DEG_TO_RAD);
    angles.push_back((180.0 + 15.0) * CH_C_DEG_TO_RAD);
    angles.push_back((180.0 + 30.0) * CH_C_DEG_TO_RAD);
    angles.push_back((180.0 + 45.0) * CH_C_DEG_TO_RAD);

    size_t NOHGT = heights.size();
    size_t NWDTH = widths.size();
    size_t NANG = angles.size();

    size_t nObs = NOHGT * NWDTH * NANG;
    size_t iObs = 1;

    std::ofstream inter("interference.txt");
    inter << "NOHGT" << std::endl;
    inter << std::setw(3) << NOHGT << std::endl;
    inter << "NANG" << std::endl;
    inter << std::setw(3) << NANG << std::endl;
    inter << "NWDTH" << std::endl;
    inter << std::setw(3) << NWDTH << std::endl;
    inter << " CLRMIN    FOOMAX    FOO       HOVALS    AVALS     WVALS" << std::endl;
    inter << " INCHES    POUNDS    POUNDS    INCHES    RADIANS   INCHES" << std::endl;

    for (size_t kWidth = 0; kWidth < NWDTH; kWidth++) {
        for (size_t jAngle = 0; jAngle < NANG; jAngle++) {
            for (size_t iHeight = 0; iHeight < NOHGT; iHeight++) {
                // --------------------------
                // Construct the Marder vehicle
                // --------------------------

                ChContactMethod contact_method = ChContactMethod::SMC;
                CollisionType chassis_collision_type = CollisionType::NONE;
                DrivelineTypeWV driveline_type = DrivelineTypeWV::AWD8;
                BrakeType brake_type = BrakeType::SIMPLE;
                EngineModelType engine_type = EngineModelType::SIMPLE_MAP;
                TransmissionModelType transmission_type = TransmissionModelType::SIMPLE_MAP;

                GD250 gd250;
                gd250.SetContactMethod(contact_method);
                gd250.SetBrakeType(brake_type);
                gd250.SetChassisCollisionType(chassis_collision_type);
                gd250.SetTireType(tire_model);
                gd250.SetKinematicMode(true);
                gd250.SetLowRangeDriveline(true);
                gd250.SetInitFwdVel(target_speed * mph_to_ms);

                ////gd250.SetChassisFixed(true);
                ////gd250.CreateTrack(false);

                // Disable gravity in this simulation
                ////gd250.GetSystem()->Set_G_acc(ChVector<>(0, 0, 0));

                // Control steering type (enable crossdrive capability)
                ////gd250.GetDriveline()->SetGyrationMode(true);

                // ------------------------------------------------
                // Initialize the vehicle at the specified position
                // ------------------------------------------------
                gd250.SetInitPosition(ChCoordsys<>(initLoc, initRot));
                gd250.Initialize();

                gd250.LockAxleDifferential(-1, true);
                gd250.LockCentralDifferential(-1, true);
                gd250.SetChassisVisualizationType(VisualizationType::NONE);
                gd250.SetSuspensionVisualizationType(VisualizationType::PRIMITIVES);
                gd250.SetSteeringVisualizationType(VisualizationType::PRIMITIVES);
                gd250.SetWheelVisualizationType(VisualizationType::NONE);
                gd250.SetTireVisualizationType(VisualizationType::MESH);

                // --------------------------------------------------
                // Control internal collisions and contact monitoring
                // --------------------------------------------------

                // Enable contact on all tracked vehicle parts, except the left sprocket
                ////gd250.GetVehicle().SetCollide(TrackedCollisionFlag::ALL & (~TrackedCollisionFlag::SPROCKET_LEFT));

                // Disable contact for all tracked vehicle parts
                ////gd250.GetVehicle().SetCollide(TrackedCollisionFlag::NONE);

                // Disable all contacts for vehicle chassis (if chassis collision was defined)
                ////gd250.GetVehicle().SetChassisCollide(false);

                // Disable only contact between chassis and track shoes (if chassis collision was defined)
                ////gd250.GetVehicle().SetChassisVehicleCollide(false);

                // Monitor internal contacts for the chassis, left sprocket, left idler, and first shoe on the left
                // track.
                ////gd250.GetVehicle().MonitorContacts(TrackedCollisionFlag::CHASSIS |
                /// TrackedCollisionFlag::SPROCKET_LEFT | /                        TrackedCollisionFlag::SHOES_LEFT |
                /// TrackedCollisionFlag::IDLER_LEFT);

                // Monitor only contacts involving the chassis.
                // gd250.GetVehicle().MonitorContacts(TrackedCollisionFlag::CHASSIS);

                // Collect contact information.
                // If enabled, number of contacts and local contact point locations are collected for all
                // monitored parts.  Data can be written to a file by invoking ChTrackedVehicle::WriteContacts().
                ////gd250.GetVehicle().SetContactCollection(true);

                // under belly points to estimate vehicle/ground interference
                std::vector<ChVector<double> > bellyPts;
                bellyPts.push_back(ChVector<>(0.8, 0, 0.1));
                bellyPts.push_back(ChVector<>(0, 0, 0));
                bellyPts.push_back(ChVector<>(-2.4, 0, 0));
                bellyPts.push_back(ChVector<>(-2.8, 0, 0));
                bellyPts.push_back(ChVector<>(-3.22, 0, 0.1));
                std::vector<ChFunction_Recorder> clearance;
                clearance.resize(bellyPts.size());

                // ------------------
                // Create the terrain
                // ------------------

                ChContactMaterialData minfo;
                minfo.mu = 0.9f;
                minfo.cr = 0.7f;
                minfo.Y = 2e7f;

                // Create the ground
                double base_height = 0.0;
                float friction_coef = 1.0f;
                double aa = CH_C_RAD_TO_DEG * angles[jAngle];
                double obl = inchToMeters * widths[kWidth];
                double obh = inchToMeters * heights[iHeight];

                ObsModTerrain terrain(gd250.GetSystem(), base_height, friction_coef, aa, obl, obh);
                // auto terrain_mat = minfo.CreateMaterial(contact_method);
                // terrain.EnableCollisionMesh(terrain_mat, std::abs(initLoc.x()) + 5, 0.03);
                terrain.Initialize(ObsModTerrain::VisualisationType::MESH);
                xpos_max = terrain.GetXObstacleEnd() + 7.0;

                // Create the driver
                auto path = ChBezierCurve::read(vehicle::GetDataFile(path_file));
                ChPathFollowerDriver driver(gd250.GetVehicle(), vehicle::GetDataFile(steering_controller_file),
                                            vehicle::GetDataFile(speed_controller_file), path, "my_path", 0.0);
                driver.Initialize();

                // Create the vehicle Irrlicht application
                auto vis = chrono_types::make_shared<ChWheeledVehicleVisualSystemIrrlicht>();
                vis->SetWindowTitle("Mercedes GD250 Obstacle Test");
                vis->SetChaseCamera(trackPoint, 10.0, 0.5);
                vis->Initialize();
                vis->AddTypicalLights();
                vis->AddSkyBox();
                vis->AddLogo();
                vis->AttachVehicle(&gd250.GetVehicle());

                // -----------------
                // Initialize output
                // -----------------

                if (!filesystem::create_directory(filesystem::path(out_dir))) {
                    std::cout << "Error creating directory " << out_dir << std::endl;
                    return 1;
                }

                if (img_output) {
                    if (!filesystem::create_directory(filesystem::path(img_dir))) {
                        std::cout << "Error creating directory " << img_dir << std::endl;
                        return 1;
                    }
                }

                // Generate JSON information with available output channels
                // gd250.GetVehicle().ExportComponentList(out_dir + "/component_list.json");

                // ------------------------------
                // Solver and integrator settings
                // ------------------------------

                // Cannot use HHT + MKL with NSC contact
                if (contact_method == ChContactMethod::NSC) {
                    use_mkl = false;
                }

#ifndef CHRONO_PARDISO_MKL
                // Cannot use HHT + PardisoMKL if Chrono::PardisoMKL not available
                use_mkl = false;
#endif

                if (use_mkl) {
#ifdef CHRONO_PARDISO_MKL
                    auto mkl_solver = chrono_types::make_shared<ChSolverPardisoMKL>();
                    mkl_solver->LockSparsityPattern(true);
                    gd250.GetSystem()->SetSolver(mkl_solver);

                    gd250.GetSystem()->SetTimestepperType(ChTimestepper::Type::HHT);
                    auto integrator = std::static_pointer_cast<ChTimestepperHHT>(gd250.GetSystem()->GetTimestepper());
                    integrator->SetAlpha(-0.2);
                    integrator->SetMaxiters(50);
                    integrator->SetAbsTolerances(1e-4, 1e2);
                    integrator->SetMode(ChTimestepperHHT::ACCELERATION);
                    integrator->SetStepControl(false);
                    integrator->SetModifiedNewton(false);
                    integrator->SetScaling(true);
                    integrator->SetVerbose(true);
#endif
                } else {
                    auto solver = chrono_types::make_shared<ChSolverBB>();
                    solver->SetMaxIterations(120);
                    solver->SetOmega(0.8);
                    solver->SetSharpnessLambda(1.0);
                    gd250.GetSystem()->SetSolver(solver);

                    gd250.GetSystem()->SetMaxPenetrationRecoverySpeed(1.5);
                    gd250.GetSystem()->SetMinBounceSpeed(2.0);
                }

                // ---------------
                // Simulation loop
                // ---------------

                // Number of simulation steps between two 3D view render frames
                int render_steps = (int)std::ceil(render_step_size / step_size);

                // Initialize simulation frame counter
                int step_number = 0;
                int render_frame = 0;

                std::ofstream kurs(out_dir + "/path.txt");

                ChTimer timer;
                timer.reset();
                timer.start();
                double sim_time = 0;
                double bail_out_time = 30.0;
                ChRunningAverage avg(100);                    // filter angine torque
                std::vector<double> engineForce;              // store obstacle related tractive force
                double effRadius = 0.328414781 + 0.06 / 2.0;  // sprocket pitch radius + track shoe thickness / 2
                double gear_ratio = 0.05;
                bool bail_out = false;
                while (vis->Run()) {
                    if (step_number % render_steps == 0) {
                        // Render scene
                        vis->BeginScene();
                        vis->Render();
                        vis->EndScene();

                        if (povray_output) {
                            char filename[100];
                            sprintf(filename, "%s/data_%03d.dat", pov_dir.c_str(), render_frame + 1);
                            utils::WriteVisualizationAssets(gd250.GetSystem(), filename);
                        }
                        if (img_output && step_number > 200) {
                            char filename[100];
                            sprintf(filename, "%s/img_%03d.jpg", img_dir.c_str(), render_frame + 1);
                            vis->WriteImageToFile(filename);
                        }
                        render_frame++;
                    }

                    double time = gd250.GetVehicle().GetChTime();
                    sim_time = time;
                    double speed = gd250.GetVehicle().GetSpeed();
                    double xpos = gd250.GetVehicle().GetPos().x();
                    double yerr = gd250.GetVehicle().GetPos().y();
                    kurs << time << "\t" << xpos << "\t" << yerr << "\t" << speed << "\t" << std::endl;
                    if (xpos >= -1.0 && xpos <= xpos_max) {
                        double eTorque =
                            avg.Add(std::static_pointer_cast<ChEngineSimpleMap>(gd250.GetVehicle().GetEngine())
                                        ->GetOutputMotorshaftTorque());
                        engineForce.push_back(eTorque * effRadius / gear_ratio);
                        for (size_t i = 0; i < bellyPts.size(); i++) {
                            ChVector<> p = gd250.GetVehicle().GetPointLocation(bellyPts[i]);
                            // GetLog() << "BellyZ(" << i << ")" << p.z() << "\n";
                            double t = terrain.GetHeight(ChVector<>(p.x(), p.y(), 0));
                            clearance[i].AddPoint(xpos, p.z() - t);
                        }
                    }
                    if (xpos > xpos_max) {
                        break;
                    }
                    if (time > bail_out_time) {
                        bail_out = true;
                        break;
                    }
                    driver.SetDesiredSpeed(ChSineStep(time, 1.0, 0.0, 2.0, target_speed));

                    // Collect output data from modules
                    DriverInputs driver_inputs = driver.GetInputs();

                    // Update modules (process inputs from other modules)
                    driver.Synchronize(time);
                    gd250.Synchronize(time, driver_inputs, terrain);
                    terrain.Synchronize(time);
                    vis->Synchronize(time, driver_inputs);

                    // Advance simulation for one timestep for all modules
                    driver.Advance(step_size);
                    gd250.Advance(step_size);
                    terrain.Advance(step_size);
                    vis->Advance(step_size);

                    xpos = gd250.GetVehicle().GetPos().x();
                    if (xpos >= xpos_max) {
                        break;
                    }

                    // Increment frame number
                    step_number++;

                    // Spin in place for real time to catch up
                    // realtime_timer.Spin(step_size);
                }

                timer.stop();
                kurs.close();

                double clearMin = 99.0;
                for (size_t i = 0; i < clearance.size(); i++) {
                    double x1, x2;
                    double vmin, vmax;
                    clearance[i].Estimate_x_range(x1, x2);
                    clearance[i].Estimate_y_range(x1, x2, vmin, vmax, 0);
                    GetLog() << "Clearance#" << i << " = " << vmin << "\n";
                    if (vmin < clearMin) {
                        clearMin = vmin;
                    }
                }

                double wallclock_time = timer.GetTimeSeconds();
                GetLog() << "Model time      = " << sim_time << " s\n";
                GetLog() << "Wall clock time = " << wallclock_time << " s\n";

                double fMax = 0.0;
                double fMean = 0.0;
                for (size_t i = 0; i < engineForce.size(); i++) {
                    if (engineForce[i] > fMax)
                        fMax = engineForce[i];
                    fMean += engineForce[i];
                }
                fMean /= double(engineForce.size());
                GetLog() << "Average Tractive Force = " << fMean << " N\n";
                GetLog() << "Max Tractive Force     = " << fMax << " N\n";
                GetLog() << "Min. Clearance         = " << clearMin << " m\n";

                double clearNogo = -19.99;
                if (bail_out) {
                    inter << std::setw(7) << std::setprecision(2) << std::fixed << clearNogo;

                } else {
                    inter << std::setw(7) << std::setprecision(2) << std::fixed << (clearMin * MetersToInch);
                }
                inter << std::setw(10) << std::setprecision(1) << std::fixed << (fMax * NewtonToLbf);
                inter << std::setw(10) << std::setprecision(1) << std::fixed << (fMean * NewtonToLbf);
                inter << std::setw(10) << std::setprecision(2) << std::fixed << heights[iHeight];
                inter << std::setw(10) << std::setprecision(2) << std::fixed << angles[jAngle];
                inter << std::setw(10) << std::setprecision(2) << std::fixed << widths[kWidth] << std::endl;
                iObs++;
            }  // height loop i
        }      // angle loop j
    }          // width loop k

    inter.close();
    return 0;
}
