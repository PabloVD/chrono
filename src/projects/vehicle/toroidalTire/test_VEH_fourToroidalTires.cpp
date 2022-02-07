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
// Authors: Antonio Recuero, Radu Serban
// =============================================================================
//
// This test creates four ANCF toroidal tires using the function MakeANCFwheel,
// which calls the class ANCFToroidal. The 4 wheels are constrained to the rims, 
// which in turn are linked to the chassis through a rev-trans joints.
// Values for the spring and damping coefficients of the secondary suspension 
// may be selected in the parameter definition section.
//
// =============================================================================

#include "chrono/solver/ChSolverMINRES.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono_irrlicht/ChIrrApp.h"

#include "chrono_vehicle/ChConfigVehicle.h"
#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/terrain/RigidTerrain.h"
#include "chrono_vehicle/wheeled_vehicle/tire/ANCFToroidalTire.h"

#ifdef CHRONO_MKL
#include "chrono_mkl/ChSolverMKL.h"
#endif

using namespace chrono;
using namespace chrono::irrlicht;
using namespace chrono::vehicle;
using namespace irr;

// =============================================================================
// Global definitions

std::shared_ptr<ChBody> BGround;
std::shared_ptr<ChBody> SimpChassis;  // Chassis body

const double spring_coef = 3e4;  // Springs and dampers for strut
const double damping_coef = 1e3;
const int num_steps = 1500;     // Number of time steps for unit test (range 1 to 4000) 550
double time_step = 0.0002;      //
const double ForVelocity = 4;   // Initial velocity of the tire. Applied to hub, nodes, and chassis

const double Lwx = 1.6;  // Wheel base
const double Lwy = 0.8;  // Track
const int NoTires = 4;   // Number of tires

const int N_DivThread = 6;
const int N_DivDiameter = 30;

const double TirePressure = 120e3;
const double TorusRadius = 0.35;
const double TorusHeight = 0.15;
const double thickness = 0.007;
const double Clearance = 0.0003;  // Initial space between tire and ground
const double GroundLoc = -(TorusRadius + TorusHeight + Clearance);  // Vertical position of the ground (for contact)

// Solver settings
enum SolverType { MINRES, MKL };
SolverType solver_type = MKL;

// =============================================================================

void MakeANCFWheel(ChSystem& my_system,
                   const ChVector<> rim_center,
                   std::shared_ptr<ChBody> & Hub_1,
                   int N_Diameter,
                   int N_Thread,
                   double TorusRadius,
                   double TorusHeight,
                   double TirePressure,
                   double ForVelocity, int Ident) {
    // Create rim for this mesh
    my_system.AddBody(Hub_1);
    Hub_1->SetIdentifier(Ident);
    Hub_1->SetBodyFixed(false);
    Hub_1->SetCollide(false);
    Hub_1->SetMass(10);
    Hub_1->SetInertiaXX(ChVector<>(0.3, 0.3, 0.3));
    Hub_1->SetPos(rim_center); 
    Hub_1->SetPos_dt(ChVector<>(ForVelocity, 0, 0));
    Hub_1->SetWvel_par(ChVector<>(0, ForVelocity / (TorusRadius + TorusHeight), 0));

    // Create the tire
    auto tire = chrono_types::make_shared<ANCFToroidalTire>("ANCF_Tire");

    tire->EnablePressure(true);
    tire->EnableContact(true);
    tire->EnableRimConnection(true);

    tire->SetDivCircumference(N_Diameter);
    tire->SetDivWidth(N_Thread);
    tire->SetAlpha(0.005);
    tire->SetHeight(TorusHeight);
    tire->SetPressure(TirePressure);
    tire->SetRimRadius(TorusRadius);
    tire->SetThickness(thickness);
    tire->Initialize(Hub_1, LEFT);

    tire->SetVisualizationType(VisualizationType::MESH);
}

// =============================================================================

int main(int argc, char* argv[]) {
    // Set path to Chrono and Chrono::Vehicle data directories
    SetChronoDataPath(CHRONO_DATA_DIR);
    vehicle::SetDataPath(CHRONO_VEHICLE_DATA_DIR);

    // Definition of the model
    ChSystemSMC my_system;

    // Body 1: Ground
    BGround = chrono_types::make_shared<ChBody>();
    my_system.AddBody(BGround);
    BGround->SetIdentifier(1);
    BGround->SetBodyFixed(true);
    BGround->SetCollide(false);
    BGround->SetMass(1);
    BGround->SetInertiaXX(ChVector<>(1, 1, 0.2));
    BGround->SetPos(ChVector<>(-2, 0, 0));  // Y = -1m
    ChQuaternion<> rot = Q_from_AngX(0.0);
    BGround->SetRot(rot);

    // Create hubs and tire meshes for 4 wheels
    auto Hub_1 = chrono_types::make_shared<ChBody>();
    auto Hub_2 = chrono_types::make_shared<ChBody>();
    auto Hub_3 = chrono_types::make_shared<ChBody>();
    auto Hub_4 = chrono_types::make_shared<ChBody>();

    ChVector<> rim_center_1(Lwx, Lwy, 0.0);
    ChVector<> rim_center_2(Lwx, -Lwy, 0.0);
    ChVector<> rim_center_3(-Lwx, -Lwy, 0.0);
    ChVector<> rim_center_4(-Lwx, Lwy, 0.0);

    MakeANCFWheel(my_system, rim_center_1, Hub_1, N_DivDiameter, N_DivThread, TorusRadius, TorusHeight, TirePressure,
                  ForVelocity, 2);
    MakeANCFWheel(my_system, rim_center_2, Hub_2, N_DivDiameter, N_DivThread, TorusRadius, TorusHeight, TirePressure,
                  ForVelocity, 3);
    MakeANCFWheel(my_system, rim_center_3, Hub_3, N_DivDiameter, N_DivThread, TorusRadius, TorusHeight, TirePressure,
                  ForVelocity, 4);
    MakeANCFWheel(my_system, rim_center_4, Hub_4, N_DivDiameter, N_DivThread, TorusRadius, TorusHeight, TirePressure,
                  ForVelocity, 5);

    SimpChassis = chrono_types::make_shared<ChBody>();
    SimpChassis->SetCollide(false);
    SimpChassis->SetMass(1500);
    my_system.AddBody(SimpChassis);

    // Visualization
    auto mobjmesh = chrono_types::make_shared<ChObjShapeFile>();
    mobjmesh->SetFilename(GetChronoDataFile("vehicle/hmmwv/hmmwv_chassis_simple.obj").c_str());
    SimpChassis->AddAsset(mobjmesh);
    SimpChassis->SetPos(ChVector<>(0, 0, 0));
    SimpChassis->SetPos_dt(ChVector<>(ForVelocity, 0, 0));
    SimpChassis->SetBodyFixed(false);

    // Create joints between chassis and hubs
    auto RevTr_1 = chrono_types::make_shared<ChLinkRevoluteTranslational>();
    my_system.AddLink(RevTr_1);
    RevTr_1->Initialize(Hub_1, SimpChassis, false, ChVector<>(Lwx, Lwy, 0.0), ChVector<>(0, 1, 0),
                        ChVector<>(Lwx, Lwy, 0.1), ChVector<>(0, 0, 1), ChVector<>(1, 0, 0), true);
    auto RevTr_2 = chrono_types::make_shared<ChLinkRevoluteTranslational>();
    my_system.AddLink(RevTr_2);
    RevTr_2->Initialize(Hub_2, SimpChassis, false, ChVector<>(Lwx, -Lwy, 0.0), ChVector<>(0, 1, 0),
                        ChVector<>(Lwx, -Lwy, 0.1), ChVector<>(0, 0, 1), ChVector<>(1, 0, 0), true);
    auto RevTr_3 = chrono_types::make_shared<ChLinkRevoluteTranslational>();
    my_system.AddLink(RevTr_3);
    RevTr_3->Initialize(Hub_3, SimpChassis, false, ChVector<>(-Lwx, -Lwy, 0.0), ChVector<>(0, 1, 0),
                        ChVector<>(-Lwx, -Lwy, 0.1), ChVector<>(0, 0, 1), ChVector<>(1, 0, 0), true);
    auto RevTr_4 = chrono_types::make_shared<ChLinkRevoluteTranslational>();
    my_system.AddLink(RevTr_4);
    RevTr_4->Initialize(Hub_4, SimpChassis, false, ChVector<>(-Lwx, Lwy, 0.0), ChVector<>(0, 1, 0),
                        ChVector<>(-Lwx, Lwy, 0.1), ChVector<>(0, 0, 1), ChVector<>(1, 0, 0), true);

    // Spring and damper for secondary suspension: True position vectors are relative
    auto spring1 = chrono_types::make_shared<ChLinkSpring>();
    spring1->Initialize(Hub_1, SimpChassis, true, ChVector<>(0, 0, 0), ChVector<>(Lwx, Lwy, 0.2), true);
    spring1->Set_SpringK(spring_coef);
    spring1->Set_SpringR(damping_coef);
    my_system.AddLink(spring1);

    auto spring2 = chrono_types::make_shared<ChLinkSpring>();
    spring2->Initialize(Hub_2, SimpChassis, true, ChVector<>(0, 0, 0), ChVector<>(Lwx, -Lwy, 0.2), true);
    spring2->Set_SpringK(spring_coef);
    spring2->Set_SpringR(damping_coef);
    my_system.AddLink(spring2);

    auto spring3 = chrono_types::make_shared<ChLinkSpring>();
    spring3->Initialize(Hub_3, SimpChassis, true, ChVector<>(0, 0, 0), ChVector<>(-Lwx, -Lwy, 0.2), true);
    spring3->Set_SpringK(spring_coef);
    spring3->Set_SpringR(damping_coef);
    my_system.AddLink(spring3);

    auto spring4 = chrono_types::make_shared<ChLinkSpring>();
    spring4->Initialize(Hub_4, SimpChassis, true, ChVector<>(0, 0, 0), ChVector<>(-Lwx, Lwy, 0.2), true);
    spring4->Set_SpringK(spring_coef);
    spring4->Set_SpringR(damping_coef);
    my_system.AddLink(spring4);

    // Create the terrain
    // ------------------

    double terrain_height = -TorusRadius - TorusHeight - Clearance;
    auto terrain = chrono_types::make_shared<RigidTerrain>(&my_system);
    auto patch = terrain->AddPatch(ChCoordsys<>(ChVector<>(0, 0, terrain_height - 5), QUNIT), ChVector<>(60, 5, 10));
    patch->SetContactFrictionCoefficient(0.9f);
    patch->SetContactRestitutionCoefficient(0.01f);
    patch->SetContactMaterialProperties(2e7f, 0.3f);
    patch->SetTexture(vehicle::GetDataFile("terrain/textures/tile4.jpg"), 200, 4);
    terrain->Initialize();

    my_system.Set_G_acc(ChVector<>(0, 0, -9.81));

    // Complete system setup
    // ---------------------

    my_system.SetupInitial();
    my_system.Setup();
    my_system.Update();

    // Solver and integrator settings
    // ------------------------------

    if (solver_type == MKL) {
#ifndef CHRONO_MKL
        solver_type = MINRES;
#endif
    }

    switch (solver_type) {
        case MINRES: {
            GetLog() << "Using MINRES solver\n";
            my_system.SetSolverType(ChSolver::Type::MINRES);
            auto minres_solver = std::static_pointer_cast<ChSolverMINRES>(my_system.GetSolver());
            ////minres_solver->SetDiagonalPreconditioning(true);
            my_system.SetSolverWarmStarting(true);
            my_system.SetMaxItersSolverSpeed(500);
            my_system.SetTolForce(1e-5);
            break;
        }
        case MKL: {
#ifdef CHRONO_MKL
            GetLog() << "Using MKL solver\n";
            auto mkl_solver = chrono_types::make_shared<ChSolverMKL<>>();
            mkl_solver->SetSparsityPatternLock(true);
            my_system.SetSolver(mkl_solver);
#endif
            break;
        }
    }

    my_system.SetTimestepperType(ChTimestepper::Type::HHT);
    auto mystepper = std::static_pointer_cast<ChTimestepperHHT>(my_system.GetTimestepper());
    mystepper->SetAlpha(-0.3);  // Important for convergence
    mystepper->SetMaxiters(11);
    mystepper->SetAbsTolerances(5e-06, 5e-03);
    mystepper->SetMode(ChTimestepperHHT::POSITION);
    mystepper->SetScaling(false);
    mystepper->SetVerbose(true);
    mystepper->SetRequiredSuccessfulSteps(2);
    mystepper->SetMaxItersSuccess(7);


    // Create the Irrlicht app
    // -----------------------

    ChIrrApp application(&my_system, L"ANCF Rolling Tire", core::dimension2d<u32>(1080, 800), false);
    application.AddLogo();
    application.AddSkyBox();
    application.AddTypicalLights();
    application.AddCamera(core::vector3df(0.5f, 0.5f, 1.15f),   // camera location
                          core::vector3df(0.65f, 0.0f, 0.0f));  // "look at" location

    application.AssetBindAll();
    application.AssetUpdateAll();

    // Perform the simulation
    // ----------------------

    chrono::GetLog() << "\n\nREADME\n\n"
                     << " - Press SPACE to start dynamic simulation \n";

    application.SetPaused(false); // at beginning, no analysis is running..
    application.SetStepManage(true);
    application.SetTimestep(time_step);
    application.SetTryRealtime(false);

    utils::CSV_writer out("\t");
    out.stream().setf(std::ios::scientific | std::ios::showpos);
    out.stream().precision(7);

    int AccuNoIterations = 0;
    double start = std::clock();

    while (application.GetDevice()->run()) {
        application.BeginScene();
        application.DrawAll();
        application.DoStep();
        application.EndScene();
        if (!application.GetPaused()) {
            std::cout << "Time t = " << my_system.GetChTime() << "s \n";
            // AccuNoIterations += mystepper->GetNumIterations();
            printf("Vertical position of Tires:      %12.4e       %12.4e       %12.4e       %12.4e  Chassis   \n",
                   Hub_1->GetPos().z(), Hub_2->GetPos().z(), Hub_3->GetPos().z(), Hub_4->GetPos().z());

            printf("Longitudinal position of Tires:      %12.4e       %12.4e       %12.4e       %12.4e  Chassis  ",
                   Hub_1->GetPos().x(), Hub_2->GetPos().x(), Hub_3->GetPos().x(), Hub_4->GetPos().x());
            out << my_system.GetChTime() << Hub_1->GetPos().x() << Hub_1->GetPos().y() << Hub_1->GetPos().z()
                << Hub_2->GetPos().x() << Hub_2->GetPos().y() << Hub_2->GetPos().z() << Hub_3->GetPos().x() << Hub_3->GetPos().y()
                << Hub_3->GetPos().z() << std::endl;
        }
    }

    double duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
    chrono::GetLog() << "Computation Time: " << duration;
    out.write_to_file("../VertPosRim.txt");

    return 0;
}
