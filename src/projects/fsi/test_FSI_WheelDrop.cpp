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
// Author: Wei Hu
// =============================================================================

// General Includes
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <valarray>
#include <string>
#include <sstream>
#include <cmath>

// Chrono includes
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsGeometry.h"
#include "chrono/assets/ChBoxShape.h"
#include "chrono/core/ChTransform.h"

// Chrono fsi includes
#include "chrono_fsi/ChSystemFsi.h"
#include "chrono_fsi/utils/ChUtilsGeneratorFsi.h"

#include "chrono/physics/ChInertiaUtils.h"
#include "chrono/assets/ChTriangleMeshShape.h"
#include "chrono/geometry/ChTriangleMeshConnected.h"

#include "chrono/ChConfig.h"
#include "chrono/core/ChStream.h"
#include "chrono/physics/ChLinkMotorRotationAngle.h"
#include "chrono/physics/ChLinkMotorRotationSpeed.h"
#include "chrono/physics/ChLinkMotorRotationTorque.h"
#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono_thirdparty/filesystem/path.h"

// Chrono namespaces
using namespace chrono;
using namespace collision;
using namespace chrono::fsi;
using namespace chrono::geometry;

// Output directories and settings
const std::string out_dir = GetChronoOutputPath() + "FSI_Rover_Single_Tire/";
std::string demo_dir;

// Save data as csv files to see the results off-line using Paraview
bool save_output = true;

double smalldis = 1.0e-9;

/// Dimension of the space domain
double bxDim = 1.0 + smalldis;
double byDim = 1.0 + smalldis;
double bzDim = 0.2 + smalldis;

/// Dimension of the terrain domain
double fxDim = 1.0 + smalldis;
double fyDim = 1.0 + smalldis;
double fzDim = 0.1 + smalldis;

// Size of the wheel
double wheel_length;
double wheel_radius;
double wheel_slip;

// linear actuator and angular actuator
auto actuator = chrono_types::make_shared<ChLinkLinActuator>();
auto motor = chrono_types::make_shared<ChLinkMotorRotationAngle>();

//------------------------------------------------------------------
// Function to save wheel to Paraview VTK files
//------------------------------------------------------------------
void WritewheelVTK(ChSystemSMC& mphysicalSystem, 
                  int this_frame,
                  std::shared_ptr<fsi::SimParams> paramsH) {
    auto body = mphysicalSystem.Get_bodylist()[1];
    ChFrame<> body_ref_frame = body->GetFrame_REF_to_abs();
    ChVector<> body_pos = body_ref_frame.GetPos();//body->GetPos();
    ChQuaternion<> body_rot = body_ref_frame.GetRot();//body->GetRot(); 

    auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
    std::string obj_path = "./hmmwv_tire_coarse_closed_small.obj";
    double scale_ratio = 1.0;
    mmesh->LoadWavefrontMesh(obj_path, false, true);
    mmesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(scale_ratio));     // scale to a different size
    mmesh->RepairDuplicateVertexes(1e-9);                                 // if meshes are not watertight                 

    double mmass;
    ChVector<> mcog;
    ChMatrix33<> minertia;
    mmesh->ComputeMassProperties(true, mmass, mcog, minertia);
    mmesh->Transform(body_pos, ChMatrix33<>(body_rot));  // rotate the mesh based on the orientation of body

    // char filename[4096];
    /*if(1==0){// save to obj file
        sprintf(filename, "%s/hmmwv_tire_%d.obj", paramsH->demo_dir, this_frame);
        std::vector<geometry::ChTriangleMeshConnected> meshes = { *mmesh };
        geometry::ChTriangleMeshConnected::WriteWavefront(filename, meshes);
    }
    if(1==1){// save to vtk file
        sprintf(filename, "%s/hmmwv_tire_%d.vtk", paramsH->demo_dir, this_frame);
        std::ofstream file;
        file.open(filename);
        file << "# vtk DataFile Version 2.0" << std::endl;
        file << "VTK from simulation" << std::endl;
        file << "ASCII" << std::endl;
        file << "DATASET UNSTRUCTURED_GRID" << std::endl;
        int nv = mmesh->getCoordsVertices().size();
        file << "POINTS " << nv << " " << "float" << std::endl;
        for (auto& v : mmesh->getCoordsVertices()) { 
            file << v.x() << " " << v.y() << " " << v.z() << std::endl;
        }
        int nf = mmesh->getIndicesVertexes().size();
        file << "CELLS " << nf << " " << 4*nf << std::endl;
        for (auto& f : mmesh->getIndicesVertexes()) {
            file <<  "3 " << f.x()  << " " << f.y() << " " << f.z()  << std::endl; 
        }
        file << "CELL_TYPES " << nf << std::endl;
        for (auto& f : mmesh->getIndicesVertexes()) {
            file <<  "5 " << std::endl; 
        }
        file.close();
    }*/

}

//------------------------------------------------------------------
// Function to save the paraview files
//------------------------------------------------------------------
void SaveParaViewFiles(ChSystemFsi& myFsiSystem,
                       ChSystemSMC& mphysicalSystem,
                       std::shared_ptr<fsi::SimParams> paramsH,
                       int this_frame,
                       double mTime,
                       std::shared_ptr<ChBody> wheel) {
    // Simulation steps between two output frames
    // int out_steps = (int)ceil((1.0 / paramsH->dT) / paramsH->out_fps);

    // Simulation time between two output frames
    double frame_time = 1.0 / paramsH->out_fps;

    // Output data to files
    if (save_output && std::abs(mTime - (this_frame)*frame_time) < 1e-9) {
        // save particles to cvs files
        myFsiSystem.PrintParticleToFile(demo_dir);
        // save rigid bodies to vtk files
        char SaveAsRigidObjVTK[256];
        static int RigidCounter = 0;
        snprintf(SaveAsRigidObjVTK, sizeof(char) * 256, (demo_dir + "/wheel.%d.vtk").c_str(), RigidCounter);
        WritewheelVTK(mphysicalSystem, this_frame, paramsH);
        RigidCounter++;
        std::cout << "\n--------------------------------\n" << std::endl;
        std::cout << "------------ Output Frame:   " << this_frame << std::endl;
        std::cout << "------------ Sim Time:       " << mTime << " (s)\n" <<std::endl;
        std::cout << "--------------------------------\n" << std::endl;
    }
}

//------------------------------------------------------------------
// Function to creat BCE particles from a mesh
//------------------------------------------------------------------
void CreateMeshMarkers(std::shared_ptr<geometry::ChTriangleMeshConnected> mesh,
                       double delta,
                       std::vector<ChVector<>>& point_cloud) {
    mesh->RepairDuplicateVertexes(1e-9);  // if meshes are not watertight

    ChVector<> minV = mesh->m_vertices[0];
    ChVector<> maxV = mesh->m_vertices[0];
    ChVector<> currV = mesh->m_vertices[0];
    for (unsigned int i = 1; i < mesh->m_vertices.size(); ++i) {
        currV = mesh->m_vertices[i];
        if (minV.x() > currV.x())
            minV.x() = currV.x();
        if (minV.y() > currV.y())
            minV.y() = currV.y();
        if (minV.z() > currV.z())
            minV.z() = currV.z();
        if (maxV.x() < currV.x())
            maxV.x() = currV.x();
        if (maxV.y() < currV.y())
            maxV.y() = currV.y();
        if (maxV.z() < currV.z())
            maxV.z() = currV.z();
    }
    ////printf("start coords: %f, %f, %f\n", minV.x(), minV.y(), minV.z());
    ////printf("end coords: %f, %f, %f\n", maxV.x(), maxV.y(), maxV.z());

    const double EPSI = 1e-6;

    ChVector<> ray_origin;
    for (double x = minV.x(); x < maxV.x(); x += delta) {
        ray_origin.x() = x + 1e-9;
        for (double y = minV.y(); y < maxV.y(); y += delta) {
            ray_origin.y() = y + 1e-9;
            for (double z = minV.z(); z < maxV.z(); z += delta) {
                ray_origin.z() = z + 1e-9;

                ChVector<> ray_dir[2] = {ChVector<>(5, 0.5, 0.25), ChVector<>(-3, 0.7, 10)};
                int intersectCounter[2] = {0, 0};

                for (unsigned int i = 0; i < mesh->m_face_v_indices.size(); ++i) {
                    auto& t_face = mesh->m_face_v_indices[i];
                    auto& v1 = mesh->m_vertices[t_face.x()];
                    auto& v2 = mesh->m_vertices[t_face.y()];
                    auto& v3 = mesh->m_vertices[t_face.z()];

                    // Find vectors for two edges sharing V1
                    auto edge1 = v2 - v1;
                    auto edge2 = v3 - v1;

                    bool t_inter[2] = {false, false};

                    for (unsigned int j = 0; j < 2; j++) {
                        // Begin calculating determinant - also used to calculate uu parameter
                        auto pvec = Vcross(ray_dir[j], edge2);
                        // if determinant is near zero, ray is parallel to plane of triangle
                        double det = Vdot(edge1, pvec);
                        // NOT CULLING
                        if (det > -EPSI && det < EPSI) {
                            t_inter[j] = false;
                            continue;
                        }
                        double inv_det = 1.0 / det;

                        // calculate distance from V1 to ray origin
                        auto tvec = ray_origin - v1;

                        // Calculate uu parameter and test bound
                        double uu = Vdot(tvec, pvec) * inv_det;
                        // The intersection lies outside of the triangle
                        if (uu < 0.0 || uu > 1.0) {
                            t_inter[j] = false;
                            continue;
                        }

                        // Prepare to test vv parameter
                        auto qvec = Vcross(tvec, edge1);

                        // Calculate vv parameter and test bound
                        double vv = Vdot(ray_dir[j], qvec) * inv_det;
                        // The intersection lies outside of the triangle
                        if (vv < 0.0 || ((uu + vv) > 1.0)) {
                            t_inter[j] = false;
                            continue;
                        }

                        double tt = Vdot(edge2, qvec) * inv_det;
                        if (tt > EPSI) {  // ray intersection
                            t_inter[j] = true;
                            continue;
                        }

                        // No hit, no win
                        t_inter[j] = false;
                    }

                    intersectCounter[0] += t_inter[0] ? 1 : 0;
                    intersectCounter[1] += t_inter[1] ? 1 : 0;
                }

                if (((intersectCounter[0] % 2) == 1) && ((intersectCounter[1] % 2) == 1))  // inside mesh
                    point_cloud.push_back(ChVector<>(x, y, z));
            }
        }
    }
}

//------------------------------------------------------------------
// Create the objects of the MBD system. Rigid bodies, and if FSI, 
// their BCE representation are created and added to the systems
//------------------------------------------------------------------
void CreateSolidPhase(ChSystemSMC& mphysicalSystem,
                      ChSystemFsi& myFsiSystem,
                      std::shared_ptr<fsi::SimParams> paramsH) {
    // Set gravity to the rigid body system in chrono
    ChVector<> gravity = ChVector<>(paramsH->gravity.x, paramsH->gravity.y, paramsH->gravity.z);
    mphysicalSystem.Set_G_acc(gravity);

    // Set common material Properties
    auto mysurfmaterial = chrono_types::make_shared<ChMaterialSurfaceSMC>();
    mysurfmaterial->SetYoungModulus(1e8);
    mysurfmaterial->SetFriction(0.9f);
    mysurfmaterial->SetRestitution(0.4f);
    mysurfmaterial->SetAdhesion(0);

    // -----------------------------------------------------
    // Create a container -- always FIRST body in the system
    // -----------------------------------------------------
    auto ground = chrono_types::make_shared<ChBodyEasyBox>(100, 100, 0.02, 1000, false, true, mysurfmaterial);
    ground->SetPos(ChVector<>(0.0, 0.0, 0.0));
    ground->SetCollide(true);
    mphysicalSystem.AddBody(ground);

    // Create the geometry of the boundaries
    double initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;

    /// Bottom wall
    ChVector<> size_XY(bxDim / 2 + 3 * initSpace0, byDim / 2 + 0 * initSpace0, 2 * initSpace0);
    ChVector<> pos_zn(0, 0, -3 * initSpace0);
    ChVector<> pos_zp(0, 0, bzDim + 2 * initSpace0);

    /// left and right Wall
    ChVector<> size_YZ(2 * initSpace0, byDim / 2 + 0 * initSpace0, bzDim / 2);
    ChVector<> pos_xp(bxDim / 2 + initSpace0, 0.0, bzDim / 2 + 0 * initSpace0);
    ChVector<> pos_xn(-bxDim / 2 - 3 * initSpace0, 0.0, bzDim / 2 + 0 * initSpace0);

    /// Front and back Wall
    ChVector<> size_XZ(bxDim / 2 + 3 * initSpace0, 2 * initSpace0, bzDim / 2);
    ChVector<> pos_yp(0, byDim / 2 + initSpace0, bzDim / 2 + 0 * initSpace0);
    ChVector<> pos_yn(0, -byDim / 2 - 3 * initSpace0, bzDim / 2 + 0 * initSpace0);

    // Add BCE particles attached on the walls into FSI system
    // myFsiSystem.AddBceBox(paramsH, ground, pos_zp, QUNIT, size_XY, 12);
    myFsiSystem.AddBceBox(paramsH, ground, pos_zn, QUNIT, size_XY, 12);
    myFsiSystem.AddBceBox(paramsH, ground, pos_xp, QUNIT, size_YZ, 23);
    myFsiSystem.AddBceBox(paramsH, ground, pos_xn, QUNIT, size_YZ, 23);
    // myFsiSystem.AddBceBox(paramsH, ground, pos_yp, QUNIT, size_XZ, 13);
    // myFsiSystem.AddBceBox(paramsH, ground, pos_yn, QUNIT, size_XZ, 13);


    // -----------------------------------------------------
    // Create the wheel -- always SECOND body in the system
    // -----------------------------------------------------
    // load mesh from obj file
    auto mmesh = chrono_types::make_shared<ChTriangleMeshConnected>();
    std::string obj_path = GetChronoDataFile ("vehicle/hmmwv/hmmwv_tire_coarse_closed.obj");
    double scale_ratio = 1.0;
    mmesh->LoadWavefrontMesh(obj_path, false, true);
    // mmesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(body_rot));       // rotate the mesh if needed
    mmesh->Transform(ChVector<>(0, 0, 0), ChMatrix33<>(scale_ratio));     // scale to a different size
    mmesh->RepairDuplicateVertexes(1e-9);                                 // if meshes are not watertight                 

    // compute mass inertia from mesh
    double mmass;
    ChVector<> mcog;
    ChMatrix33<> minertia;
    double mdensity = paramsH->bodyDensity;
    mmesh->ComputeMassProperties(true, mmass, mcog, minertia);
    ChMatrix33<> principal_inertia_rot;
    ChVector<> principal_I;
    ChInertiaUtils::PrincipalInertia(minertia, principal_I, principal_inertia_rot);
    mcog = ChVector<>(0.0, 0.0, 0.0);

    // set the abs orientation, position and velocity
    auto wheel = chrono_types::make_shared<ChBodyAuxRef>();
    ChQuaternion<> Body_rot = Q_from_Euler123(ChVector<double>(0, 0, 0));
    ChVector<> Body_pos = ChVector<>(paramsH->bodyIniPosX, paramsH->bodyIniPosY, paramsH->bodyIniPosZ + fzDim) ;
    ChVector<> Body_vel = ChVector<>(paramsH->bodyIniVelX, paramsH->bodyIniVelY, paramsH->bodyIniVelZ);

    // Set the COG coordinates to barycenter, without displacing the REF reference.
    // Make the COG frame a principal frame.
    wheel->SetFrame_COG_to_REF(ChFrame<>(mcog, principal_inertia_rot));

    // Set inertia
    wheel->SetMass(paramsH->bodyMass * 1.0 / 2.0);
    wheel->SetInertiaXX(mdensity * principal_I);
    wheel->SetPos_dt(Body_vel);
    wheel->SetWvel_loc(ChVector<>(0.0, 0.0, 0.0)); // set an initial anular velocity (rad/s)
    
    // Set the absolute position of the body:
    wheel->SetFrame_REF_to_abs(ChFrame<>(ChVector<>(Body_pos),ChQuaternion<>(Body_rot)));                              
    mphysicalSystem.AddBody(wheel);

    wheel->SetBodyFixed(false);
    wheel->GetCollisionModel()->ClearModel();
    wheel->GetCollisionModel()->AddTriangleMesh(mysurfmaterial, mmesh,false, false, VNULL, ChMatrix33<>(1), 0.005);
    wheel->GetCollisionModel()->BuildModel();
    wheel->SetCollide(false);

    // auto masset_mesh = chrono_types::make_shared<ChTriangleMeshShape>();
    // masset_mesh->SetMesh(mmesh);
    // masset_mesh->SetBackfaceCull(true);
    // wheel->AddAsset(masset_mesh);

    // Add this body to the FSI system
    std::vector<ChVector<>> BCE_par_rock;
    CreateMeshMarkers(mmesh, (double)initSpace0, BCE_par_rock);
    myFsiSystem.AddBceFromPoints(paramsH, wheel, BCE_par_rock, ChVector<>(0.0), QUNIT);
    myFsiSystem.AddFsiBody(wheel);
    

    // -----------------------------------------------------
    // Create the chassis -- always THIRD body in the system
    // -----------------------------------------------------
    // Initially, the chassis is fixed to ground.
    // It is released after the settling phase.
    auto chassis = chrono_types::make_shared<ChBody>();
    // chassis->SetIdentifier(Id_chassis);
    chassis->SetMass(paramsH->bodyMass * 1.0 / 2.0);
    chassis->SetPos(wheel->GetPos());
    chassis->SetCollide(false);
    chassis->SetBodyFixed(false);

    // Add geometry of the chassis.
    chassis->GetCollisionModel()->ClearModel();
    chrono::utils::AddBoxGeometry(chassis.get(), mysurfmaterial, ChVector<>(0.1, 0.1, 0.1), ChVector<>(0, 0, 0));
    chassis->GetCollisionModel()->BuildModel();
    mphysicalSystem.AddBody(chassis);

    // ---------------------------------------------------
    // Create the axle -- always FOURTH body in the system
    // ---------------------------------------------------
    auto axle = chrono_types::make_shared<ChBody>();
    // axle->SetIdentifier(Id_axle);
    axle->SetMass(paramsH->bodyMass * 1.0 / 2.0);
    axle->SetPos(wheel->GetPos());
    axle->SetCollide(false);
    axle->SetBodyFixed(false);
    // Add geometry of the axle.
    axle->GetCollisionModel()->ClearModel();
    chrono::utils::AddSphereGeometry(axle.get(), mysurfmaterial, 0.5, ChVector<>(0, 0, 0));
    axle->GetCollisionModel()->BuildModel();
    mphysicalSystem.AddBody(axle);


    // =============================================================================
    // Connect the chassis to the containing bin (ground) through a translational
    // joint and create a linear actuator.
    // =============================================================================
    auto prismatic1 = chrono_types::make_shared<ChLinkLockPrismatic>();
    prismatic1->Initialize(ground, chassis, ChCoordsys<>(chassis->GetPos(), Q_from_AngY(CH_C_PI_2)));
    prismatic1->SetName("prismatic_chassis_ground");
    mphysicalSystem.AddLink(prismatic1);

    double angVel = paramsH->bodyIniAngVel;
    double velocity = angVel * wheel_radius * (1.0 - wheel_slip);
    auto actuator_fun = chrono_types::make_shared<ChFunction_Ramp>(0.0, velocity);

    // auto actuator = chrono_types::make_shared<ChLinkLinActuator>();
    actuator->Initialize(ground, chassis, false, ChCoordsys<>(chassis->GetPos(), QUNIT),
                         ChCoordsys<>(chassis->GetPos() + ChVector<>(1, 0, 0), QUNIT));
    actuator->SetName("actuator");
    actuator->Set_lin_offset(1);
    actuator->Set_dist_funct(actuator_fun);
    mphysicalSystem.AddLink(actuator);

    // =============================================================================
    // Connect the axle to the chassis through a vertical translational
    // joint.
    // =============================================================================
    auto prismatic2 = chrono_types::make_shared<ChLinkLockPrismatic>();
    prismatic2->Initialize(chassis, axle, ChCoordsys<>(chassis->GetPos(), QUNIT));
    prismatic2->SetName("prismatic_axle_chassis");
    mphysicalSystem.AddLink(prismatic2);

    // =============================================================================
    // Connect the wheel to the axle through a engine joint.
    // =============================================================================
    // auto motor = chrono_types::make_shared<ChLinkMotorRotationAngle>();
    motor->SetName("engine_wheel_axle");
    motor->Initialize(wheel, axle, ChFrame<>(wheel->GetPos(), chrono::Q_from_AngAxis(-CH_C_PI / 2.0, ChVector<>(1, 0, 0))));
    motor->SetAngleFunction(chrono_types::make_shared<ChFunction_Ramp>(0, angVel));
    mphysicalSystem.AddLink(motor);
}



// =============================================================================
int main(int argc, char* argv[]) {
    // Create a physics system and an FSI system
    ChSystemSMC mphysicalSystem;
    ChSystemFsi myFsiSystem(mphysicalSystem);

    // Get the pointer to the system parameter and use a JSON file to fill it out with the user parameters
    std::shared_ptr<fsi::SimParams> paramsH = myFsiSystem.GetSimParams();

    // Use the default input file or you may enter your input parameters as a command line argument
    std::string inputJson = GetChronoDataFile("fsi/input_json/test_FSI_WheelDrop.json");
    myFsiSystem.SetSimParameter(inputJson, paramsH, ChVector<>(bxDim, byDim, bzDim));

    /// Set SPH discretization type, consistent or inconsistent
    myFsiSystem.SetDiscreType(false, false);

    /// Set wall boundary condition
    myFsiSystem.SetWallBC(BceVersion::ORIGINAL);

    /// Reset the domain size
    bxDim = paramsH->boxDimX;
    byDim = paramsH->boxDimY;
    bzDim = paramsH->boxDimZ;
    /// Reset the terrain size
    fxDim = paramsH->fluidDimX;
    fyDim = paramsH->fluidDimY;
    fzDim = paramsH->fluidDimZ;

    // Size of the dropping wheel
    wheel_radius = paramsH->bodyRad;
    wheel_length = paramsH->bodyLength;

    /// Setup the solver based on the input value of the prameters
    myFsiSystem.SetFluidDynamics(paramsH->fluid_dynamic_type);

    // Set up the periodic boundary condition (if not, set relative larger values)
    double initSpace0 = paramsH->MULT_INITSPACE * paramsH->HSML;
    ChVector<> cMin(-bxDim / 2 * 10, -byDim / 2 - 0.5 * initSpace0, -bzDim * 10 - 10 * initSpace0);
    ChVector<> cMax( bxDim / 2 * 10,  byDim / 2 + 0.5 * initSpace0,  bzDim * 10 + 10 * initSpace0);
    myFsiSystem.SetBoundaries(cMin, cMax, paramsH);

    /// Setup sub doamins for a faster neighbor particle searching
    myFsiSystem.SetSubDomain(paramsH);

    /// Setup the output directory for FSI data
    myFsiSystem.SetFsiOutputDir(paramsH, demo_dir, out_dir, inputJson.c_str());

    /// Set FSI information output
    myFsiSystem.SetFsiInfoOutput(false);

    /// Set simulation data output length
    myFsiSystem.SetOutputLength(0);


    // Create Fluid region and discretize with SPH particles
    ChVector<> boxCenter(0.0, 0.0, fzDim / 2);
    ChVector<> boxHalfDim(fxDim / 2, fyDim / 2, fzDim / 2);

    // Use a chrono sampler to create a bucket of points
    chrono::utils::GridSampler<> sampler(initSpace0);
    std::vector<ChVector<>> points = sampler.SampleBox(boxCenter, boxHalfDim);

    // Add fluid particles from the sampler points to the FSI system
    int numPart = (int)points.size();
    for (int i = 0; i < numPart; i++) {
        double pre_ini = paramsH->rho0 * abs(paramsH->gravity.z) * (-points[i].z() + fzDim);
        myFsiSystem.AddSphMarker(points[i], paramsH->rho0, 0, paramsH->mu0, paramsH->HSML, -1,
                                 ChVector<>(0),         // initial velocity
                                 ChVector<>(-pre_ini),  // tauxxyyzz
                                 ChVector<>(0)          // tauxyxzyz
        );
    }
    myFsiSystem.AddRefArray(0, (int)numPart, -1, -1);


    // Create Solid region and attach BCE SPH particles
    CreateSolidPhase(mphysicalSystem, myFsiSystem, paramsH);

    // Construction of the FSI system must be finalized before running
    myFsiSystem.Finalize();

    // Set up integrator for the multi-body dynamics system
    // mphysicalSystem.SetTimestepperType(ChTimestepper::Type::HHT);
    // auto mystepper = std::static_pointer_cast<ChTimestepperHHT>(mphysicalSystem.GetTimestepper());
    // mystepper->SetAlpha(-0.2);
    // mystepper->SetMaxiters(1000);
    // mystepper->SetAbsTolerances(1e-6);
    // mystepper->SetMode(ChTimestepperHHT::ACCELERATION);
    // mystepper->SetScaling(true);

    auto wheel = mphysicalSystem.Get_bodylist()[1];
    ChVector<> force = actuator->Get_react_force();
    ChVector<> torque = motor->Get_react_torque();
    ChVector<> w_pos = wheel->GetPos();
    ChVector<> w_vel = wheel->GetPos_dt();
    ChVector<> angvel = wheel->GetWvel_loc();

    /// write the infomation into file
    std::ofstream myFile;

    // Start the simulation
    double time = 0;
    int stepEnd = int(paramsH->tFinal / paramsH->dT);
    double TIMING_sta = clock();
    for (int tStep = 0; tStep < stepEnd + 1; tStep++) {
        printf("\nstep : %d, time= : %f (s) \n", tStep, time);
        double frame_time = 1.0 / paramsH->out_fps;
        int this_frame = (int)floor((time + 1e-9) / frame_time);

        // Get the wheel from the FSI system and Save data of the simulation
        // std::vector<std::shared_ptr<ChBody>>& FSI_Bodies = myFsiSystem.GetFsiBodies();

        // auto wheel = FSI_Bodies[0];
        SaveParaViewFiles(myFsiSystem, mphysicalSystem, paramsH, this_frame, time, wheel);

        // Get the infomation of the wheel
        force = actuator->Get_react_force();
        torque = motor->Get_react_torque();
        w_pos = wheel->GetPos();
        w_vel = wheel->GetPos_dt();
        angvel = wheel->GetWvel_loc();
        printf("wheel position = %f,%f,%f\n", w_pos.x(), w_pos.y(), w_pos.z());
        printf("wheel velocity = %f,%f,%f\n", w_vel.x(), w_vel.y(), w_vel.z());
        printf("wheel ang velocity = %f,%f,%f\n", angvel.x(), angvel.y(), angvel.z());
        printf("draw-bar pull = %f,%f,%f\n", force.x(), force.y(), force.z());
        printf("wheel torque = %f,%f,%f\n", torque.x(), torque.y(), torque.z());

        myFile.open(demo_dir + "/results.txt", std::ios::app);
        myFile << time << "\t"
            << w_pos.x() << "\t" << w_pos.y() << "\t" << w_pos.z() << "\t"
            << w_vel.x() << "\t" << w_vel.y() << "\t" << w_vel.z() << "\t"
            << angvel.x() << "\t" << angvel.y() << "\t" << angvel.z() << "\t"
            << force.x() << "\t" << force.y() << "\t" << force.z() << "\t"
            << torque.x() << "\t" << torque.y() << "\t" << torque.z() << "\n";
        myFile.close();
        
        // Call the FSI solver
        myFsiSystem.DoStepDynamics_FSI();
        time += paramsH->dT;
    }

    // Total computational cost
    double TIMING_end = (clock() - TIMING_sta) / (double)CLOCKS_PER_SEC;
    printf("\nSimulation Finished in %f (s)\n", TIMING_end);

    return 0;
}

