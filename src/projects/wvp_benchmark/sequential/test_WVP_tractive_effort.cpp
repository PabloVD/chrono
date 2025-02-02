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
// Authors: Radu Serban, Asher Elmquist
// =============================================================================
//
// Sample test program for sequential WVP simulation.
//
// The vehicle reference frame has Z up, X towards the front of the vehicle, and
// Y pointing to the left.
// All units SI.
//
// =============================================================================

#define USE_IRRLICHT


#include "chrono_vehicle/ChConfigVehicle.h"
#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/terrain/RigidTerrain.h"

#ifdef USE_IRRLICHT
#include "chrono_vehicle/driver/ChInteractiveDriverIRR.h"
#endif

#include "chrono_vehicle/wheeled_vehicle/ChWheeledVehicleVisualSystemIrrlicht.h"
#include "chrono_vehicle/driver/ChPathFollowerDriver.h"
#include "chrono_vehicle/terrain/SCMTerrain.h"
#include "chrono_models/vehicle/wvp/WVP_FollowerDataDriver.h"
#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono/motion_functions/ChFunction.h"

#include "chrono_models/vehicle/wvp/WVP.h"

#include "chrono_thirdparty/filesystem/path.h"

#include <chrono>
#include <thread>
#include <math.h>

using namespace chrono;
using namespace chrono::vehicle;
using namespace chrono::vehicle::wvp;

// =============================================================================

// Initial vehicle location and orientation
ChVector<> initLoc(-180, 0, 1.0);
ChQuaternion<> initRot(1, 0, 0, 0);

ChVector<>gravity(0,0,-9.81);
ChVector<>slopegravity(0,0,-9.81);

// Visualization type for vehicle parts (PRIMITIVES, MESH, or NONE)
VisualizationType chassis_vis_type = VisualizationType::NONE;
VisualizationType suspension_vis_type = VisualizationType::PRIMITIVES;
VisualizationType steering_vis_type = VisualizationType::PRIMITIVES;
VisualizationType wheel_vis_type = VisualizationType::NONE;
VisualizationType tire_vis_type = VisualizationType::PRIMITIVES;

// Type of tire model (RIGID, FIALA, PAC89)
TireModelType tire_model = TireModelType::RIGID;

// Point on chassis tracked by the camera
ChVector<> trackPoint(0.0, 0.0, 1.75);

double terrainLength = 400;
double terrainWidth = 5;

// Simulation step sizes
double step_size =1e-3;
double tire_step_size = step_size;

// Simulation end time
double tend = 15;

// Time interval between two render frames
double render_step_size = 1.0 / 50;  // FPS = 50
double output_step_size = 1e-2;

//output directory
const std::string out_dir = "../WVP_TRACTIVE_EFFORT";
const std::string pov_dir = out_dir + "/POVRAY";
bool povray_output = false;
bool data_output = true;

//vehicle driver inputs
// Desired vehicle speed (m/s)
double mph_to_ms = 0.44704;
double target_speed = 5*mph_to_ms;

// std::string path_file("paths/NATO_double_lane_change.txt");
std::string path_file("paths/straightOrigin.txt");
std::string steering_controller_file("wvp/SteeringController.json");
std::string speed_controller_file("wvp/SpeedControllerAgrPI.json");

std::string output_file_name("tractive");

bool LtR = false;

// =============================================================================

int main(int argc, char* argv[]) {
    //send in args: filename time speed LtR(or RtL)
    //read from args

    //arg1: simulationg length, arg2=max force to be ramped to over t=10 to t=20 and hold
    double fullForce = 0;
    if(argc > 2){
        tend = atof(argv[1]);
        fullForce = atof(argv[2]);
        output_file_name += std::to_string((int)fullForce);
    }
    else{
        std::cout<<"Standard Setup"<<std::endl;
    }


    // --------------
    // Create systems
    // --------------

    // Create the vehicle, set parameters, and initialize
    WVP wvp;
    wvp.SetChassisFixed(false);
    wvp.SetInitPosition(ChCoordsys<>(initLoc, initRot));
    wvp.SetTireType(tire_model);
    wvp.SetTireStepSize(tire_step_size);
    wvp.Initialize();

    wvp.GetSystem()->Set_G_acc(gravity);

    wvp.SetChassisVisualizationType(chassis_vis_type);
    wvp.SetSuspensionVisualizationType(suspension_vis_type);
    wvp.SetSteeringVisualizationType(steering_vis_type);
    wvp.SetWheelVisualizationType(wheel_vis_type);
    wvp.SetTireVisualizationType(tire_vis_type);

    std::cout << "Vehicle mass:               " << wvp.GetVehicle().GetMass() << std::endl;

    // ------------------
    // Create the terrain
    // ------------------

    //Deformable terrain properties (LETE sand)
    double Kphi = 5301e3;
    double Kc = 102e3;
    double n = 0.793;
    double c = 1.3e3;
    double phi = 31.1;
    double K = 1.2e-2;
    double E_elastic = 2e8;
    double damping = 3e4;

    float mu = 0.8f;
    float restitution = 0.01f;
    float E = 2e7f;
    float nu = 0.3f;
    double depth = 10;

    auto terrain = new SCMTerrain(wvp.GetSystem());
    terrain->SetPlane(ChCoordsys<>(VNULL, Q_from_AngX(CH_C_PI_2)));
    terrain->SetSoilParameters(Kphi,Kc,n,c,phi,K,E_elastic,damping);
    // terrain->SetPlotType(vehicle::SCMTerrain::PLOT_PRESSURE_YIELD,0,30000.2);
    terrain->SetPlotType(vehicle::SCMTerrain::PLOT_SINKAGE, 0, 0.15);

    terrain->Initialize(terrainLength, terrainWidth, 0.1);

    //create the driver
    auto path = ChBezierCurve::read(vehicle::GetDataFile(path_file));
    ChPathFollowerDriver driver(wvp.GetVehicle(), vehicle::GetDataFile(steering_controller_file),
                                vehicle::GetDataFile(speed_controller_file), path, "my_path", target_speed);
    driver.Initialize();



    //create the drawbar force
    ChFunction_Recorder forceFunct;
    forceFunct.AddPoint(0,0);
    forceFunct.AddPoint(10,0);
    forceFunct.AddPoint(20.0,-fullForce);

    // forceFunct.AddPoint(0,0);
    // forceFunct.AddPoint(3,0);
    // forceFunct.AddPoint(13,-fullForce);

    auto drawForce = chrono_types::make_shared<ChForce>();
    wvp.GetVehicle().GetChassisBody()->AddForce(drawForce);
    drawForce->SetMode(ChForce::ForceType::FORCE); // force or torque
    drawForce->SetFrame(ChForce::ReferenceFrame::BODY);
    drawForce->SetAlign(ChForce::AlignmentFrame::WORLD_DIR);
    drawForce->SetVrelpoint(wvp.GetVehicle().GetCOMFrame().GetPos());
    drawForce->SetF_x(chrono_types::make_shared<ChFunction_Recorder>(forceFunct));

    drawForce->SetF_y(chrono_types::make_shared<ChFunction_Const>(0));
    drawForce->SetF_z(chrono_types::make_shared<ChFunction_Const>(0));




#ifdef USE_IRRLICHT
    // -------------------------------------
    // Create the vehicle Irrlicht interface
    // Create the driver system
    // -------------------------------------

    auto vis = chrono_types::make_shared<ChWheeledVehicleVisualSystemIrrlicht>();
    vis->SetWindowTitle("WVP sequential test");
    vis->SetChaseCamera(trackPoint, 6.0, 0.5);
    vis->Initialize();
    vis->AddTypicalLights();
    vis->AddSkyBox();
    vis->AddLogo();
    vis->AttachVehicle(&wvp.GetVehicle());

    // Visualization of controller points (sentinel & target)
    irr::scene::IMeshSceneNode* ballS = vis->GetSceneManager()->addSphereSceneNode(0.1f);
    irr::scene::IMeshSceneNode* ballT = vis->GetSceneManager()->addSphereSceneNode(0.1f);
    ballS->getMaterial(0).EmissiveColor = irr::video::SColor(0, 255, 0, 0);
    ballT->getMaterial(0).EmissiveColor = irr::video::SColor(0, 0, 255, 0);
#endif

    // -------------
    // Prepare output
    // -------------

    if (data_output || povray_output) {
        if (!filesystem::create_directory(filesystem::path(out_dir))) {
            std::cout << "Error creating directory " << out_dir << std::endl;
            return 1;
        }
    }
    if (povray_output) {
        if (!filesystem::create_directory(filesystem::path(pov_dir))) {
            std::cout << "Error creating directory " << pov_dir << std::endl;
            return 1;
        }
        driver.ExportPathPovray(out_dir);
    }

    utils::CSV_writer csv("\t");
    csv.stream().setf(std::ios::scientific | std::ios::showpos);
    csv.stream().precision(6);

    // Number of simulation steps between two 3D view render frames
    int render_steps = (int)std::ceil(render_step_size / step_size);

    // Number of simulation steps between two output frames
    int output_steps = (int)std::ceil(output_step_size / step_size);

    // ---------------
    // Simulation loop
    // ---------------

    int step_number = 0;
    int render_frame = 0;

    double time = 0;

    //output headings for the saved data file
    csv << "time";
    csv << "throttle";
    csv << "MotorSpeed";
    csv << "VehicleSpeed";
    csv << "EngineTorque";

    for(int i=0;i<4;i++){
        csv << "WheelAngVelX";
        csv << "WheelAngVelY";
        csv << "WheelAngVelZ";
    }

    for(int i=0;i<4;i++){
        csv << "WheelLongSlip";
    }

    csv << "DrawBarForce";
    csv << std::endl;


#ifdef USE_IRRLICHT
    while (vis->Run()) {


        //path visualization
        const ChVector<>& pS = driver.GetSteeringController().GetSentinelLocation();
        const ChVector<>& pT = driver.GetSteeringController().GetTargetLocation();
        ballS->setPosition(irr::core::vector3df((irr::f32)pS.x(), (irr::f32)pS.y(), (irr::f32)pS.z()));
        ballT->setPosition(irr::core::vector3df((irr::f32)pT.x(), (irr::f32)pT.y(), (irr::f32)pT.z()));

        /*std::cout<<"Target:\t"<<(irr::f32)pT.x()<<",\t "<<(irr::f32)pT.y()<<",\t "<<(irr::f32)pT.z()<<std::endl;
        std::cout<<"Vehicle:\t"<<wvp.GetVehicle().GetChassisBody()->GetPos().x()
          <<",\t "<<wvp.GetVehicle().GetChassisBody()->GetPos().y()<<",\t "
          <<wvp.GetVehicle().GetChassisBody()->GetPos().z()<<std::endl;*/

        // Render scene
        if (step_number % render_steps == 0) {
            vis->BeginScene();
            vis->Render();
            vis->EndScene();
        }
#else
    while(wvp.GetSystem()->GetChTime() < tend){
#endif

        time = wvp.GetSystem()->GetChTime();

        if (time >= 10.0 && time < 10.1)
            wvp.GetSystem()->Set_G_acc(slopegravity);

        // Driver inputs
        DriverInputs driver_inputs = driver.GetInputs();

        // Update modules (process inputs from other modules)
        driver.Synchronize(time);
        terrain->Synchronize(time);
        wvp.Synchronize(time, driver_inputs, *terrain);
#ifdef USE_IRRLICHT
        vis->Synchronize(time, driver_inputs);
#endif

        // Advance simulation for one timestep for all modules

        driver.Advance(step_size);
        terrain->Advance(step_size);
        wvp.Advance(step_size);
#ifdef USE_IRRLICHT
        vis->Advance(step_size);
#endif

        if (povray_output && step_number % render_steps == 0) {
            char filename[100];
            sprintf(filename, "%s/data_%03d.dat", pov_dir.c_str(), render_frame + 1);
            utils::WriteVisualizationAssets(wvp.GetSystem(), filename);
            render_frame++;
        }

        if (data_output && step_number % output_steps == 0) {
            // output time to check simulation is running
            std::cout << time << std::endl;

            csv << driver_inputs.m_throttle;
            csv << wvp.GetVehicle().GetEngine()->GetMotorSpeed();
            csv << wvp.GetVehicle().GetSpeed();
            csv << wvp.GetVehicle().GetEngine()->GetOutputMotorshaftTorque();

            for (int axle = 0; axle < 2; axle++) {
                csv << wvp.GetVehicle().GetSpindleAngVel(axle, LEFT);
                csv << wvp.GetVehicle().GetSpindleAngVel(axle, RIGHT);
            }

            for (auto& axle : wvp.GetVehicle().GetAxles()) {
                for (auto& wheel : axle->GetWheels()) {
                    csv << wheel->GetTire()->GetLongitudinalSlip();
                }
            }

            csv << forceFunct.Get_y(time);

            csv << std::endl;
        }

        // std::cout<<time<<std::endl;
        // Increment frame number
        step_number++;
    }

    if (data_output) {
        csv.write_to_file(out_dir + "/" + output_file_name + ".dat");
    }

    return 0;
}
