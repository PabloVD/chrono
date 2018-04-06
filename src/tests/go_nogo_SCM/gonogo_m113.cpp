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
// 
//
// =============================================================================

#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

#include "chrono/ChConfig.h"
#include "chrono/core/ChFileutils.h"
#include "chrono/core/ChMathematics.h"
#include "chrono/geometry/ChLineBezier.h"
#include "chrono/solver/ChIterativeSolver.h"
#include "chrono/utils/ChFilters.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono_vehicle/ChDriver.h"
#include "chrono_vehicle/ChVehicle.h"
#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/terrain/SCMDeformableTerrain.h"
#include "chrono_vehicle/utils/ChVehiclePath.h"
#include "chrono_vehicle/utils/ChSteeringController.h"

#include "chrono_models/vehicle/m113a/M113a_SimplePowertrain.h"
#include "chrono_models/vehicle/m113a/M113a_Vehicle.h"

////#undef CHRONO_IRRLICHT
#ifdef CHRONO_IRRLICHT
#include "chrono_vehicle/tracked_vehicle/utils/ChTrackedVehicleIrrApp.h"
#endif

#include "chrono_thirdparty/SimpleOpt/SimpleOpt.h"
#include "chrono_thirdparty/filesystem/path.h"
#include "chrono_thirdparty/filesystem/resolver.h"

using namespace chrono;
using namespace chrono::vehicle;
using namespace chrono::vehicle::m113;

using std::cout;
using std::endl;

// -----------------------------------------------------------------------------
// Specification of terrain
// -----------------------------------------------------------------------------

// Slope scaling factor (in degrees)
double max_slope = 40;

// Patch half-dimensions
double hdimX = 100;//// 250;
double hdimY = 2.5;

// Initial number of divisions per unit (m)
double factor = 6;

// Vehicle horizontal offset
double horizontal_offset = 6;
double horizontal_pos = hdimX - horizontal_offset;

// Initial vehicle position, orientation, and forward velocity
ChVector<> initLoc(-horizontal_pos, 0, 0.8);
ChQuaternion<> initRot(1, 0, 0, 0);
double initSpeed = 0;

// Point on chassis tracked by the camera (Irrlicht only)
ChVector<> trackPoint(-1.0, 0.0, 0.0);

// -----------------------------------------------------------------------------
// Timed events
// -----------------------------------------------------------------------------

// Total simulation duration.
double time_end = 50;

// Delay before starting the engine
double time_start_engine = 0.25;

// Delay before throttle reaches maximum (linear ramp)
double delay_max_throttle = 0.5;
double time_max_throttle = time_start_engine + delay_max_throttle;

// Delays before checking for slow-down and steady-state
double filter_interval = 3.0;
double delay_start_check_slow = 5.0;
double delay_start_check_steady = 5.0;
double time_start_check_slow = time_max_throttle + delay_start_check_slow;
double time_start_check_steady = time_max_throttle + delay_start_check_steady;

// Time when terrain is pitched (rotate gravity)
double time_pitch = time_start_engine;

// -----------------------------------------------------------------------------
// Simulation parameters
// -----------------------------------------------------------------------------

// Simulation step size
double time_step = 1e-3;

// Output
bool output = true;
double output_frequency = 100.0;
std::string out_dir = "../GONOGO_M113_SCM";

// =============================================================================

void ShowUsage(const std::string& name);
bool GetProblemSpecs(int argc, char** argv, std::string& file, int& line, bool& copy);

// =============================================================================

class GONOGO_Driver : public chrono::vehicle::ChDriver {
  public:
    GONOGO_Driver(chrono::vehicle::ChVehicle& vehicle,          // associated vehicle
                  std::shared_ptr<chrono::ChBezierCurve> path,  // target path
                  bool render_path,                             // add curve visualization asset?
                  double time_start,                            // time throttle start
                  double time_max                               // time throttle max
    );

    void SetGains(double Kp, double Ki, double Kd) { m_steeringPID.SetGains(Kp, Ki, Kd); }
    void SetLookAheadDistance(double dist) { m_steeringPID.SetLookAheadDistance(dist); }

    void Reset() { m_steeringPID.Reset(m_vehicle); }

    virtual void Synchronize(double time) override;
    virtual void Advance(double step) override;

    void ExportPathPovray(const std::string& out_dir);

  private:
    chrono::vehicle::ChPathSteeringController m_steeringPID;
    double m_start;
    double m_end;
};

// =============================================================================

int main(int argc, char* argv[]) {
    GetLog() << "Copyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";

    // ----------------------------
    // Parse command line arguments
    // ----------------------------

    std::string input_file = "";  // Name of input file
    int line_number = 0;          // Line of inputs
    bool copy = true;             // Copy input file?

    // Extract arguments
    if (!GetProblemSpecs(argc, argv, input_file, line_number, copy)) {
        return 1;
    }

    // Check that input file exists
    filesystem::path inpath(input_file);
    if (!inpath.exists()) {
        cout << "Input file " << input_file << " does not exist" << endl;
        return 1;
    } else if (!inpath.is_file()) {
        cout << "Input file " << input_file << " is not a regular file" << endl;
        return 1;
    }

    // Check that a line number was specified
    if (line_number <= 0) {
        cout << "Incorrect line number." << endl;
        return 1;
    }

    // ----------------
    // Parse input file
    // ----------------

    // Extract the filename, the basename, and extension of the input file
    std::string filename = inpath.filename();
    std::string stem = inpath.stem();
    std::string extension = inpath.extension();

    // Open input file
    std::ifstream ifile;
    ifile.open(input_file.c_str());

    std::string line;
    for (int i = 0; i < line_number; i++) {
        if (!std::getline(ifile, line) || line.length() == 0) {
            cout << "Incorrect line number." << endl;
            return 1;
        }
    }
    ifile.clear();
    ifile.seekg(0);

    // Extract input data
    int model_index;
    double slope_val, saturation_val;
    double Bekker_n, Bekker_Kphi, Bekker_Kc, Mohr_coh, Mohr_phi, Janosi_k;
    std::istringstream iss(line);
    iss >> model_index >> slope_val >> saturation_val >> Bekker_n >> Bekker_Kphi >> Bekker_Kc >> Mohr_coh >> Mohr_phi >> Janosi_k;
    
    double slope_deg = slope_val * max_slope;
    double slope = slope_deg * (CH_C_PI / 180);
    Bekker_Kphi *= 1000;
    Bekker_Kc *= 1000;
    Mohr_coh *= 1000;
    Janosi_k *= 1e-2;

    cout << "Set up" << endl;
    cout << "  File:          " << input_file << "  Line: " << line_number << endl;
    cout << "  Parameters:    " << model_index << " " << slope_val << " " << saturation_val << endl;
    cout << "  Slope:         " << slope_deg << endl;
    cout << "  Bekker Kphi:   " << Bekker_Kphi << endl;
    cout << "  Bekker Kc:     " << Bekker_Kc << endl;
    cout << "  Bekker n:      " << Bekker_n << endl;
    cout << "  Mohr cohesion: " << Mohr_coh << endl;
    cout << "  Mohr phi:      " << Mohr_phi << endl;
    cout << "  Janosi k:      " << Janosi_k << endl;

    // ---------------------------------
    // Create output directory and files
    // ---------------------------------

    std::ofstream ofile;
    std::string del("  ");

    if (output) {
        if (ChFileutils::MakeDirectory(out_dir.c_str()) < 0) {
            cout << "Error creating directory " << out_dir << endl;
            return 1;
        }

        out_dir += "/" + stem;

        if (ChFileutils::MakeDirectory(out_dir.c_str()) < 0) {
            cout << "Error creating directory " << out_dir << endl;
            return 1;
        }

        // Copy input file to output directory
        if (copy) {
            std::ofstream dst(out_dir + "/" + filename, std::ios::binary);
            dst << ifile.rdbuf();
        }

        // Open the output file stream
        char buf[10];
        std::sprintf(buf, "%03d", line_number);
        ofile.open(out_dir + "/results_" + std::string(buf) + ".out", std::ios::out);
    }

    // -------------
    // Create system
    // -------------

    // Prepare rotated acceleration vector
    ChVector<> gravity(0, 0, -9.81);
    ChVector<> gravityR = ChMatrix33<>(slope, ChVector<>(0, 1, 0)) * gravity;

    ChSystemSMC* system = new ChSystemSMC();
    system->Set_G_acc(gravity);

    // system->SetMaxItersSolverSpeed(1000);
    // system->SetMaxItersSolverStab(1000);
    system->SetSolverType(ChSolver::Type::MINRES);
    system->SetSolverWarmStarting(true);
    // vehicle.GetSystem()->SetTolForce(1e-2);
    auto solver = std::static_pointer_cast<ChIterativeSolver>(system->GetSolver());
    solver->SetRecordViolation(true);

    // -------------------------------------
    // Create the vehicle
    // -------------------------------------

    M113a_Vehicle m113(false, system);

    m113.Initialize(ChCoordsys<>(initLoc, initRot));

    // Set visualization type for subsystems
    m113.SetChassisVisualizationType(VisualizationType::NONE);
    m113.SetSprocketVisualizationType(VisualizationType::MESH);
    m113.SetIdlerVisualizationType(VisualizationType::MESH);
    m113.SetRoadWheelAssemblyVisualizationType(VisualizationType::MESH);
    m113.SetRoadWheelVisualizationType(VisualizationType::MESH);
    m113.SetTrackShoeVisualizationType(VisualizationType::MESH);

    // Control steering type (enable crossdrive capability).
    m113.GetDriveline()->SetGyrationMode(true);

    // Create and initialize the powertrain system
    M113a_SimplePowertrain powertrain("Powertrain");
    powertrain.Initialize(m113.GetChassisBody(), m113.GetDriveshaft());

    // -------------------------------------
    // Create the driver
    // -------------------------------------

    double height = initLoc.z();
    auto path = StraightLinePath(ChVector<>(-2 * hdimX, 0, height), ChVector<>(200 * hdimX, 0, height));

    GONOGO_Driver driver(m113, path, false, time_start_engine, time_max_throttle);
    double look_ahead_dist = 5;
    double Kp_steering = 0.5;
    double Ki_steering = 0;
    double Kd_steering = 0;
    driver.SetLookAheadDistance(look_ahead_dist);
    driver.SetGains(Kp_steering, Ki_steering, Kd_steering);
    driver.Initialize();

    // ------------------
    // Create the terrain
    // ------------------

    // Deformable terrain properties (LETE sand)
    double Kphi = 5301e3;    // Bekker Kphi
    double Kc = 102e3;       // Bekker Kc
    double n = 0.793;        // Bekker n exponent
    double c = 1.3e3;        // Mohr cohesive limit (Pa)
    double phi = 31.1;       // Mohr friction limit (degrees)
    double K = 1.2e-2;       // Janosi shear coefficient (m)
    double E_elastic = 2e8;  // Elastic stiffness (Pa/m), before plastic yeld
    double damping = 3e4;    // Damping coefficient (Pa*s/m)

    // Mesh divisions
    int ndivX = (int)std::ceil(2 * hdimX * factor);
    int ndivY = (int)std::ceil(2 * hdimY * factor);

    SCMDeformableTerrain terrain(system);
    terrain.SetPlane(ChCoordsys<>(VNULL, Q_from_AngX(CH_C_PI_2)));
    terrain.SetSoilParametersSCM(Bekker_Kphi, Bekker_Kc, Bekker_n, Mohr_coh, Mohr_phi, K, E_elastic, damping);
    ////terrain.SetPlotType(vehicle::SCMDeformableTerrain::PLOT_PRESSURE_YELD, 0, 30000.2);
    terrain.SetPlotType(vehicle::SCMDeformableTerrain::PLOT_SINKAGE, 0, 0.15);
    terrain.Initialize(0, 2 * hdimX, 2 * hdimY, ndivX, ndivY);
    terrain.SetAutomaticRefinement(true);
    terrain.SetAutomaticRefinementResolution(0.02);

    // Enable moving patch feature
    terrain.EnableMovingPatch(m113.GetChassisBody(), ChVector<>(-2, 0, 0), 6.5, 3.5);

#ifdef CHRONO_IRRLICHT

    // ---------------------------------------
    // Create the vehicle Irrlicht application
    // ---------------------------------------

    ChTrackedVehicleIrrApp app(&m113, &powertrain, L"M113 go/no-go");
    app.SetSkyBox();
    app.AddLight(irr::core::vector3df(-100.f, -150.f, 150.f), 150, irr::video::SColorf(0.65f, 0.65f, 0.7f));
    app.AddLight(irr::core::vector3df(-100.f, +150.f, 150.f), 150, irr::video::SColorf(0.65f, 0.65f, 0.7f));
    app.AddTypicalLogo();
    app.SetChaseCamera(trackPoint, 5.0, 0.25);
    app.AssetBindAll();
    app.AssetUpdateAll();

    ////app.SetChaseCameraAngle(CH_C_PI / 5);
    ////app.EnableStats(false);
#endif

    // Save parameters and problem setup to output file
    if (output) {
        ofile << "# File: " << filename << endl;
        ofile << "# Line: " << line_number << endl;
        ofile << "# " << endl;
        ofile << "# " << line << endl;
        ofile << "# " << endl;
        ofile << "# Model index:      " << model_index << endl;
        ofile << "# Slope level:      " << slope_val << endl;
        ofile << "# Saturation level: " << saturation_val << endl;
        ofile << "# Slope (deg):      " << slope_deg << endl;
        ofile << "# Bekker Kphi:      " << Bekker_Kphi << endl;
        ofile << "# Bekker Kc:        " << Bekker_Kc << endl;
        ofile << "# Bekker n:         " << Bekker_n << endl;
        ofile << "# Mohr cohesion:    " << Mohr_coh << endl;
        ofile << "# Mohr phi:         " << Mohr_phi << endl;
        ofile << "# Janosi k:         " << Janosi_k << endl;
        ofile << "# " << endl;

        ofile.precision(7);
        ofile << std::scientific;
    }

    // ---------------
    // Simulation loop
    // ---------------

    // Inter-module communication data
    BodyStates shoe_states_left(m113.GetNumTrackShoes(LEFT));
    BodyStates shoe_states_right(m113.GetNumTrackShoes(RIGHT));
    TerrainForces shoe_forces_left(m113.GetNumTrackShoes(LEFT));
    TerrainForces shoe_forces_right(m113.GetNumTrackShoes(RIGHT));

    // Number of simulation steps between two output frames
    int output_steps = (int)std::ceil((1 / output_frequency) / time_step);

    double time = 0;
    int sim_frame = 0;
    int out_frame = 0;
    int next_out_frame = 0;
    double exec_time = 0;

    bool is_pitched = false;

    int filter_steps = (int)std::ceil(filter_interval / time_step);
    utils::ChRunningAverage fwd_vel_filter(filter_steps);
    utils::ChRunningAverage fwd_acc_filter(filter_steps);

    while (true) {
        // Rotate gravity vector
        if (!is_pitched && time > time_pitch) {
            cout << time << "    Pitch: " << gravityR.x() << " " << gravityR.y() << " " << gravityR.z() << endl;
            system->Set_G_acc(gravityR);
            is_pitched = true;
        }

        // Check if reached maximum simulation time
        if (time >= time_end) {
            if (output) {
                ofile << "# " << endl;
                ofile << "# Reached maximum time" << endl;
            }
            break;
        }

#ifdef CHRONO_IRRLICHT
        if (!app.GetDevice()->run())
            break;

        app.BeginScene(true, true, irr::video::SColor(255, 140, 161, 192));
        app.DrawAll();
#endif

        // Extract chassis state
        ChVector<> pv = m113.GetChassisBody()->GetFrame_REF_to_abs().GetPos();
        ChVector<> vv = m113.GetChassisBody()->GetFrame_REF_to_abs().GetPos_dt();
        ChVector<> av = m113.GetChassisBody()->GetFrame_REF_to_abs().GetPos_dtdt();

        // Check if vehicle at maximum position
        if (pv.x() >= horizontal_pos) {
            if (output) {
                ofile << "# " << endl;
                ofile << "# Reached maximum x position" << endl;
            }
            break;
        }

        // Filtered forward velocity and acceleration
        double fwd_vel_mean = fwd_vel_filter.Add(vv.x());
        double fwd_vel_std = fwd_vel_filter.GetStdDev();
        double fwd_acc_mean = fwd_acc_filter.Add(av.x());
        double fwd_acc_std = fwd_acc_filter.GetStdDev();

        // Check if vehicle is sliding backward
        if (pv.x() <= -horizontal_pos - 1) {
            if (output) {
                ofile << "# " << endl;
                ofile << "# Vehicle sliding backward" << endl;
            }
            break;
        }

        // Check if vehicle is slowing down
        if (time > time_start_check_slow && fwd_acc_mean < 0.1) {
            if (output) {
                ofile << "# " << endl;
                ofile << "# Vehicle slowing down" << endl;
            }
            break;
        }

        // Check if vehicle reached steady-state speed
        if (time > time_start_check_steady && std::abs(fwd_acc_mean) < 0.05 && fwd_vel_std < 0.03) {
            if (output) {
                ofile << "# " << endl;
                ofile << "# Vehicle reached steady state" << endl;
            }
            break;
        }

        // Collect output data from modules (for inter-module communication)
        double steering_input = driver.GetSteering();
        double braking_input = driver.GetBraking();
        double throttle_input = driver.GetThrottle();
        double powertrain_torque = powertrain.GetOutputTorque();
        double driveshaft_speed = m113.GetDriveshaftSpeed();
        m113.GetTrackShoeStates(LEFT, shoe_states_left);
        m113.GetTrackShoeStates(RIGHT, shoe_states_right);

        // Save output
        if (output && sim_frame == next_out_frame) {
            cout << system->GetChTime() << " pos: " << pv.x() << " vel: " << vv.x() << " Wtime: " << exec_time << endl;

            ofile << system->GetChTime() << del;
            ofile << throttle_input << del << steering_input << del;

            ofile << pv.x() << del << pv.y() << del << pv.z() << del;
            ofile << vv.x() << del << vv.y() << del << vv.z() << del;

            ofile << fwd_vel_mean << del << fwd_vel_std << del;
            ofile << fwd_acc_mean << del << fwd_acc_std << del;

            ofile << endl;

            out_frame++;
            next_out_frame += output_steps;
        }

        // Synchronize subsystems
        terrain.Synchronize(time);
        driver.Synchronize(time);
        powertrain.Synchronize(time, throttle_input, driveshaft_speed);
        m113.Synchronize(time, steering_input, braking_input, powertrain_torque, shoe_forces_left, shoe_forces_right);
#ifdef CHRONO_IRRLICHT
        app.Synchronize("", steering_input, throttle_input, braking_input);
#endif

        // Advance systems
        driver.Advance(time_step);
        powertrain.Advance(time_step);
        terrain.Advance(time_step);
        m113.Advance(time_step);
#ifdef CHRONO_IRRLICHT
        app.Advance(time_step);
#endif

        ////terrain.PrintStepStatistics(cout);

        // Update counters.
        time += time_step;
        sim_frame++;
        exec_time += system->GetTimerStep();

#ifdef CHRONO_IRRLICHT
        app.EndScene();
#endif
    }

    // Final stats
    cout << "==================================" << endl;
    cout << "Simulation time:   " << exec_time << endl;

    if (output) {
        ofile << "# " << endl;
        ofile << "# Simulation time (s): " << exec_time << endl;
        ofile.close();
    }

    return 0;
}

// =============================================================================

GONOGO_Driver::GONOGO_Driver(chrono::vehicle::ChVehicle& vehicle,
                             std::shared_ptr<chrono::ChBezierCurve> path,
                             bool render_path,
                             double time_start,
                             double time_max)
    : chrono::vehicle::ChDriver(vehicle), m_steeringPID(path, false), m_start(time_start), m_end(time_max) {
    m_steeringPID.Reset(m_vehicle);

    if (render_path) {
        auto road = std::shared_ptr<chrono::ChBody>(m_vehicle.GetSystem()->NewBody());
        road->SetBodyFixed(true);
        m_vehicle.GetSystem()->AddBody(road);

        auto path_asset = std::make_shared<chrono::ChLineShape>();
        path_asset->SetLineGeometry(std::make_shared<chrono::geometry::ChLineBezier>(m_steeringPID.GetPath()));
        path_asset->SetColor(chrono::ChColor(0.0f, 0.8f, 0.0f));
        path_asset->SetName("straight_path");
        road->AddAsset(path_asset);
    }
}

void GONOGO_Driver::Synchronize(double time) {
    m_braking = 0;
    if (time < m_start) {
        m_throttle = 0;
    } else if (time < m_end) {
        m_throttle = (time - m_start) / (m_end - m_start);
    } else {
        m_throttle = 1;
    }
}

void GONOGO_Driver::Advance(double step) {
    double out_steering = m_steeringPID.Advance(m_vehicle, step);
    chrono::ChClampValue(out_steering, -1.0, 1.0);
    m_steering = out_steering;
}

void GONOGO_Driver::ExportPathPovray(const std::string& out_dir) {
    chrono::utils::WriteCurvePovray(*m_steeringPID.GetPath(), "straight_path", out_dir, 0.04,
                                    chrono::ChColor(0.8f, 0.5f, 0.0f));
}

// =============================================================================
// ID values to identify command line arguments
enum { OPT_HELP, OPT_FILE, OPT_LINE, OPT_NO_COPY };

// Table of CSimpleOpt::Soption structures. Each entry specifies:
// - the ID for the option (returned from OptionId() during processing)
// - the option as it should appear on the command line
// - type of the option
// The last entry must be SO_END_OF_OPTIONS
CSimpleOptA::SOption g_options[] = {{OPT_FILE, "-f", SO_REQ_CMB},
                                    {OPT_LINE, "-l", SO_REQ_CMB},
                                    {OPT_NO_COPY, "--no-copy", SO_NONE},
                                    {OPT_HELP, "-?", SO_NONE},
                                    {OPT_HELP, "-h", SO_NONE},
                                    {OPT_HELP, "--help", SO_NONE},
                                    SO_END_OF_OPTIONS};

void ShowUsage(const std::string& name) {
    std::cout << "Usage: " << name << " -f=FILE_NAME -l=LINE -t=THREADS [OPTIONS]" << std::endl;
    std::cout << " -f=FILE_NAME" << std::endl;
    std::cout << "        Name of input file" << std::endl;
    std::cout << "        Each line contains a point in parameter space:" << std::endl;
    std::cout << "        slope (deg), radius (mm), density (kg/m3), coef. friction, cohesion" << std::endl;
    std::cout << " -l=LINE" << std::endl;
    std::cout << "        Line in input file" << std::endl;
    std::cout << " --no-copy" << std::endl;
    std::cout << "        Disable copying of input file to output directory" << std::endl;
    std::cout << " -? -h --help" << std::endl;
    std::cout << "        Print this message and exit." << std::endl;
    std::cout << std::endl;
}

bool GetProblemSpecs(int argc, char** argv, std::string& file, int& line, bool& copy) {
    // Create the option parser and pass it the program arguments and the array of valid options.
    CSimpleOptA args(argc, argv, g_options);

    copy = true;

    // Then loop for as long as there are arguments to be processed.
    while (args.Next()) {
        // Exit immediately if we encounter an invalid argument.
        if (args.LastError() != SO_SUCCESS) {
            std::cout << "Invalid argument: " << args.OptionText() << std::endl;
            ShowUsage(argv[0]);
            return false;
        }

        // Process the current argument.
        switch (args.OptionId()) {
            case OPT_HELP:
                ShowUsage(argv[0]);
                return false;
            case OPT_FILE:
                file = args.OptionArg();
                break;
            case OPT_LINE:
                line = std::stoi(args.OptionArg());
                break;
            case OPT_NO_COPY:
                copy = false;
                break;
        }
    }

    return true;
}
