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
// Author: Radu Serban
// =============================================================================
//
// Chrono::Vehicle + Chrono::Multicore demo program for simulating a HMMWV vehicle
// over rigid or granular material.
//
// Contact uses the SMC (penalty) formulation.
//
// The global reference frame has Z up.
// All units SI.
// =============================================================================

#include <cstdio>
#include <cmath>
#include <vector>

#include "chrono/physics/ChSystemNSC.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/ChDriver.h"
#include "SCMTerrain_Custom.h"
#include "chrono_vehicle/terrain/RigidTerrain.h"
//#include "chrono_vehicle/wheeled_vehicle/utils/ChWheeledVehicleVisualSystemIrrlicht.h"
#include "chrono_vehicle/wheeled_vehicle/ChWheeledVehicleVisualSystemIrrlicht.h"

#include "chrono_models/vehicle/hmmwv/HMMWV.h"

#include "chrono_models/robot/curiosity/Curiosity.h"
#include "chrono_vehicle/ChVehicleModelData.h"


#include "chrono_thirdparty/cxxopts/ChCLI.h"
#include "chrono_thirdparty/filesystem/path.h"

#include "DataWriter.h"
#include "CreateObjects.h"

using namespace chrono;
using namespace chrono::collision;
using namespace chrono::irrlicht;
using namespace chrono::vehicle;
using namespace chrono::vehicle::hmmwv;
using namespace chrono::geometry;
using namespace chrono::curiosity;

using std::cout;
using std::endl;

#include <torch/torch.h>
#include <torch/script.h>
// #include <torchscatter/scatter.h>
// #include <torchcluster/cluster.h>
// torch::jit::script::Module module;

// torch::Tensor tensor = torch::eye(3);
// std::cout << tensor << std::endl;

bool GetProblemSpecs(int argc,
                     char** argv,
                     std::string& terrain_dir, double& tend, double& throttle, double& steering, double& render_step_size, bool& heightmapterrain, double& initheight, bool& use_nn);

// =============================================================================
// USER SETTINGS
// =============================================================================

// -----------------------------------------------------------------------------
// Default terrain parameters
// -----------------------------------------------------------------------------

std::string terrain_dir;
double tend = 0.5;

// double terrainLength = 8.0;  // size in X direction
// double terrainWidth = 3.0;    // size in Y direction
// double terrainLength = 15.0;  // size in X direction
// double terrainWidth = 3.0;    // size in Y direction
double terrainLength = 35.0;  // size in X direction
double terrainWidth = 17.5;    // size in Y direction
// double terrainLength = 100.0;  // size in X direction
// double terrainWidth = 6.0;    // size in Y direction
// double terrainLength = 70.0;  // size in X direction
// double terrainWidth = 10.0;    // size in Y direction
double delta = 0.05;          // SCM grid spacing

double throttle=0.;
double steering=0.;
bool heightmapterrain=true;
double initheight=0.1;
//double maxheight = 0.5;
double maxheight = 1.5*35.0/40.0;
//double maxheight = 3.;

// Flag for using NN
//bool use_nn = 1;
bool use_nn = 0;

// -----------------------------------------------------------------------------
// Simulation parameters
// -----------------------------------------------------------------------------

// Simulation step size
double step_size = 2e-3;

// Time interval between two render frames (1/FPS)
// double render_step_size = 2.0 / 100;
double render_step_size=step_size;

// Point on chassis tracked by the camera
ChVector<> trackPoint(0.0, 0.0, 1.75);

// Flag for printing time stats
bool print_stats = true;

// Visualization output
bool img_output = false;

// Vertices output
bool ver_output = false;

// =============================================================================
// Rover stuff
// =============================================================================

// Specify rover chassis type (Scarecrow or FullRover)
CuriosityChassisType chassis_type = CuriosityChassisType::FullRover;

// Specify rover wheel type (RealWheel, SimpleWheel, or CylWheel)
CuriosityWheelType wheel_type = CuriosityWheelType::RealWheel;


// =============================================================================


bool GetProblemSpecs(int argc,
                     char** argv,
                     std::string& terrain_dir, double& tend, double& throttle, double& steering, double& render_step_size, bool& heightmapterrain, double& initheight, bool& use_nn) 
    {
    ChCLI cli(argv[0], "Polaris SCM terrain simulation");

    cli.AddOption<std::string>("Problem setup", "terrain_dir", "Directory with terrain specification data");
    cli.AddOption<double>("Simulation", "tend", "Simulation end time [s]", std::to_string(tend));
    cli.AddOption<double>("Simulation", "throttle", "Simulation throttle magnitude ", std::to_string(throttle));
    cli.AddOption<double>("Simulation", "steering", "Simulation steering magnitude ", std::to_string(steering));
    cli.AddOption<double>("Simulation", "render_step_size", "Simulation render and output step size ", std::to_string(render_step_size));
    cli.AddOption<double>("Simulation", "initheight", "Spawning height in meters ", std::to_string(initheight));
    cli.AddOption<bool>("Simulation", "use_nn", "Use NN for true or standard SCM for false", std::to_string(use_nn));
    if (!cli.Parse(argc, argv)) {
        cli.Help();
        return false;
    }

    try {
        terrain_dir = cli.Get("terrain_dir").as<std::string>();
    } catch (std::domain_error&) {
        cout << "\nERROR: Missing terrain specification directory!\n\n" << endl;
        heightmapterrain=false;
        // cli.Help();
        // return false;
    }
    tend = cli.GetAsType<double>("tend");
    throttle = cli.GetAsType<double>("throttle");
    steering = cli.GetAsType<double>("steering");
    render_step_size = cli.GetAsType<double>("render_step_size");
    initheight = cli.GetAsType<double>("initheight");
    use_nn = cli.GetAsType<bool>("use_nn");


    return true;
}

class MyDriver : public ChDriver {
  public:
    MyDriver(ChVehicle& vehicle, double delay) : ChDriver(vehicle), m_delay(delay) {}
    ~MyDriver() {}

    virtual void Synchronize(double time) override {
        m_throttle = 0.0;
        m_steering = 0.0;
        m_braking = 1.0;

        double eff_time = time - m_delay;

        // Do not generate any driver inputs for a duration equal to m_delay.
        if (eff_time < 0.0)
            return;

        if (eff_time > 0.0)
        { 
            //m_throttle = throttle;
            m_throttle = throttle * (std::sin(CH_C_2PI * (eff_time) / 2.) + 1.5);
            m_braking = 0.0;
            m_steering = steering * std::sin(CH_C_2PI * (eff_time) / 4.);
        }
        
        if (m_throttle > 1.)
        {
            m_throttle = 1.;
        }         
    }

  private:
    double m_delay;
};



// =============================================================================

int main(int argc, char* argv[]) {
    GetLog() << "Copyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";
    
    if (!GetProblemSpecs(argc, argv,                                 
                         terrain_dir, tend, throttle, steering, render_step_size, heightmapterrain, initheight, use_nn)) 
    {
        return 1;
    }
    // // Check input files exist
    // if (!filesystem::path(vehicle::GetDataFile(terrain_dir + "/path.txt")).exists()) {
    //     std::cout << "Input file path.txt not found in directory " << terrain_dir << std::endl;
    //     return 1;
    // }
    // Parse command line arguments
    bool verbose = true;
    bool wheel_output = true;      // save individual wheel output files
    double output_major_fps = 1.0/render_step_size;

    // Output directories
    std::string out_dir = GetChronoOutputPath() + "POLARIS_SCM";
    
    if (use_nn){
        out_dir +=  "_nn";
    }
    else{
        out_dir += "_scm";
    }
    if (heightmapterrain){

        std::string hmapstring = "hmap_";
        
        int found_in = terrain_dir.find(hmapstring)  + hmapstring.length();
        int found_end = terrain_dir.find(".png");
        std::string hmap_num = terrain_dir.substr(found_in, found_end-found_in);
        out_dir += "_" + hmapstring + hmap_num;
    }
    else{
        out_dir +=  "_flat";
    }
    std::stringstream str_throttle, str_steering, str_initheight, str_tend;
    str_throttle << std::fixed << std::setprecision(2) << throttle;
    str_steering << std::fixed << std::setprecision(2) << steering;
    str_initheight << std::fixed << std::setprecision(2) << initheight;
    str_tend << std::fixed << std::setprecision(2) << tend;
    out_dir += "_throttle_"+str_throttle.str()+"_steering_"+str_steering.str();
    out_dir += "_initheight_"+str_initheight.str();
    out_dir += "_tend_"+str_tend.str();
    const std::string img_dir = out_dir + "/IMG";


    // --------------------
    // Create the Chrono systems
    // --------------------
    ChSystemNSC sys;
    sys.SetNumThreads(std::min(8, ChOMP::GetNumProcs()));
    const ChVector<> gravity(0, 0, -9.81);
    sys.Set_G_acc(gravity);


    // ------------------
    // Create the terrain
    // ------------------

    SCMTerrain_Custom terrain(&sys, true, use_nn, 4);
    //SCMTerrain_Custom terrain(&sys, vehicle);
    // SCMDeformableTerrain terrain(system);

    terrain.SetSoilParameters(2e6,   // Bekker Kphi
                                0,     // Bekker Kc
                                1.1,   // Bekker n exponent
                                0,     // Mohr cohesive limit (Pa)
                                30,    // Mohr friction limit (degrees)
                                0.01,  // Janosi shear coefficient (m)
                                2e8,   // Elastic stiffness (Pa/m), before plastic yield
                                0       // Damping (Pa s/m), proportional to negative vertical speed (optional)
                                //3e4    // Damping (Pa s/m), proportional to negative vertical speed (optional)
    );

    ////terrain.EnableBulldozing(true);      // inflate soil at the border of the rut
    ////terrain.SetBulldozingParameters(55,   // angle of friction for erosion of displaced material at rut border
    ////                                0.8,  // displaced material vs downward pressed material.
    ////                                5,    // number of erosion refinements per timestep
    ////                                10);  // number of concentric vertex selections subject to erosion

    // Optionally, enable moving patch feature (single patch around vehicle chassis)
    // terrain.AddMovingPatch(my_hmmwv.GetChassisBody(), ChVector<>(0, 0, 0), ChVector<>(5, 3, 1));

    // Optionally, enable moving patch feature (multiple patches around each wheel)
    ////for (auto& axle : my_hmmwv.GetVehicle().GetAxles()) {
    ////    terrain.AddMovingPatch(axle->m_wheels[0]->GetSpindle(), ChVector<>(0, 0, 0), ChVector<>(1, 0.5, 1));
    ////    terrain.AddMovingPatch(axle->m_wheels[1]->GetSpindle(), ChVector<>(0, 0, 0), ChVector<>(1, 0.5, 1));
    ////}

    ////terrain.SetTexture(vehicle::GetDataFile("terrain/textures/grass.jpg"), 80, 16);
    ////terrain.SetPlotType(vehicle::SCMDeformableTerrain::PLOT_PRESSURE_YELD, 0, 30000.2);
    terrain.SetPlotType(vehicle::SCMTerrain_Custom::PLOT_SINKAGE, 0, 0.05);

    if (heightmapterrain)
    { 
     terrain.Initialize(terrain_dir,  ///< [in] filename for the height map (image file)
                    terrainLength ,                       ///< [in] terrain dimension in the X direction
                    terrainWidth,                       ///< [in] terrain dimension in the Y direction
                    0.0,                        ///< [in] minimum height (black level)
                    maxheight,                        ///< [in] maximum height (white level)
                    delta                        ///< [in] grid spacing (may be slightly decreased)
     );        
    }
    else
    {
     terrain.Initialize(terrainLength, terrainWidth, delta);    
    }

    // --------------------
    // Create the Polaris vehicle
    // --------------------

    // Initial vehicle position and orientation
    //ChCoordsys<> init_pos(ChVector<>(1.3, 0, 0.1), QUNIT);

    // First find height in the spawn point to ensure that the vehicle spawns above the floor
    double init_x = 4.-terrainLength/2.0;
    //double init_x = 0.;
    ChVector<> initLoc0(init_x, 0, initheight);
    double terrainheightspawn = terrain.GetHeight(initLoc0);
    // cout << "Height terrain in spawn point: " << terrainheightspawn << endl;

    ChVector<> initLoc(init_x, 0, initheight + terrainheightspawn);
    ChQuaternion<> initRot(1, 0, 0, 0); // Same than QUNIT
    ChCoordsys<> init_pos(initLoc, initRot);

    cout << "Create vehicle..." << endl;
    //auto vehicle = CreateVehicle(sys, init_pos);
    std::shared_ptr<Curiosity> vehicle = chrono_types::make_shared<Curiosity>(&sys, chassis_type, wheel_type);
    cout << "Llega hasta aqui" << endl;
    //Curiosity rover(&sys, chassis_type, wheel_type);
    vehicle->SetDriver(chrono_types::make_shared<CuriositySpeedDriver>(1.0, CH_C_PI));
    //vehicle.Initialize(ChFrame<>(ChVector<>(-5, -0.2, 0), Q_from_AngX(-CH_C_PI / 2)));
    cout << "Llega hasta aqui" << endl;
    vehicle->Initialize(ChFrame<>(initLoc, initRot));
    //std::shared_ptr<Curiosity> vehicle = std::make_shared<Curiosity>(rover);
    double x_max = (terrainLength/2.0 - 3.0);
    double y_max = (terrainWidth/2.0 - 3.0);

    terrain.EnterVehicle(vehicle);


    cout << "Llega hasta aqui" << endl;

    // std::string vertices_filename = out_dir +  "/vertices_" + std::to_string(0) + ".csv";
    // terrain.WriteMeshVertices(vertices_filename);

    // ---------------------------------------
    // Create the vehicle Irrlicht application
    // ---------------------------------------
    auto vis = chrono_types::make_shared<ChWheeledVehicleVisualSystemIrrlicht>();
    vis->SetWindowTitle("Rover SCM");
    vis->SetChaseCamera(trackPoint, 6.0, 0.5);
    vis->Initialize();
    vis->AddLightDirectional();
    vis->AddSkyBox();
    vis->AddLogo();
    //vis->AttachVehicle(vehicle.get());
    //vis->AttachVehicle(vehicle);

    // --------------------
    // Create driver system
    // --------------------
    // MyDriver driver(*vehicle, 0.5);
    // driver.Initialize();

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

    // DataWriterVehicle data_writer(&sys, vehicle, terrain);
    // data_writer.SetVerbose(verbose);
    // data_writer.SetMBSOutput(wheel_output);
    // data_writer.Initialize(out_dir, output_major_fps, step_size);
    cout << "Simulation output data saved in: " << out_dir << endl;
    cout << "===============================================================================" << endl;

    // ---------------
    // Simulation loop
    // ---------------
    //std::cout << "Total vehicle mass: " << vehicle->GetMass() << std::endl;


    // Number of simulation steps between two 3D view render frames
    int render_steps = (int)std::ceil(render_step_size / step_size);

    // Initialize simulation frame counter
    int step_number = 0;
    int render_frame = 0;
    double t = 0;
    //while (t < tend) {


    ChTimer timer_tot, timer_sync, timer_vis, timer_advance;
    

    while (vis->Run()) {

        timer_sync.reset();
        timer_vis.reset();
        timer_advance.reset();
        timer_tot.reset();

        timer_tot.start();
    

        // const auto& veh_loc = vehicle->GetPos();
        // std::cout<<"veh_loc ="<<veh_loc<<std::endl;
        // Stop before end of patch
        // if (veh_loc.x() > x_max || veh_loc.y() > y_max )
        //     break;

        timer_vis.start();

        // Render scene
        vis->Run();
        vis->BeginScene();
        vis->Render();
        tools::drawColorbar(vis.get(), 0, 0.1, "Sinkage", 30);
        vis->EndScene();

        timer_vis.stop();

        // if (ver_output)
        //     data_writer.Process(step_number, t);
        
        if (step_number % render_steps == 0) {
            if (ver_output)
            {   
            std::string vertices_filename = out_dir +  "/vertices_" + std::to_string(render_frame) + ".csv";
            if (step_number==0)
            {
             terrain.WriteMeshVertices(vertices_filename);
            }
            else
            {
             //terrain.WriteMeshVerticesinz(vertices_filename);
             terrain.WriteModifiedMeshVertices(vertices_filename);
            }
            }
            if (img_output)
            {
            //char filename[100];
            //sprintf(filename, "%s/img_%03d.jpg", img_dir.c_str(), render_frame + 1);
            std::string  filename = img_dir + "/img_"+ std::to_string(render_frame) +".jpg";
            vis->WriteImageToFile(filename);
            }
            render_frame++;
        }

        timer_sync.start();

        // // Driver inputs
        //DriverInputs driver_inputs = driver.GetInputs();
        DriverInputs driver_inputs = {0.0, 0.0, 0.0};

        // // Update modules
        //driver.Synchronize(t);
        terrain.Synchronize(t);
        //vehicle->Synchronize(t, driver_inputs, terrain);
        vis->Synchronize(t, driver_inputs);

        timer_sync.stop();
        timer_advance.start();

        // Advance dynamics
        sys.DoStepDynamics(step_size);
        vis->Advance(step_size);
        t += step_size;

        timer_advance.stop();
        timer_tot.stop();

        // Increment frame number
        step_number++;

        // Pablo
        if (print_stats){
            terrain.PrintStepStatistics(cout);
            std::cout << "Visualization time by step (ms): " << 1e3 * timer_vis() << std::endl;
            std::cout << "Synchronization time by step (ms): " << 1e3 * timer_sync() << std::endl;
            std::cout << "Advance time by step (ms): " << 1e3 * timer_advance() << std::endl;
            std::cout << "Total time by step (ms): " << 1e3 * timer_tot() << std::endl;
            
        }

        if (t>=tend)
            break;
    }

    return 0;
}

