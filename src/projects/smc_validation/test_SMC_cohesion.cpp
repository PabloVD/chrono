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
// Authors: Cecily Sunday, Radu Serban
// =============================================================================
//
//  Two spheres are initially held together by adhesion, in the absence of gravity
//  Gravity is turned on first with F_gravity < F_adhesion, then with
//  F_gravity > F_adhesion. The spheres should stay in contact during case 1
//  but seperate during case 2.
//
// =============================================================================

#include "./test_SMC.h"

std::ofstream CreateDataFile() {
    const std::string fname = GetChronoOutputPath() + "/dat.csv";
    std::ofstream dat(fname, std::ofstream::out);

    // Write header
    dat << "force_model, gcase, body_id, time, radius, collision_map,"
        << "pos_x, pos_y, pos_z,"
        << "rot_0, rot_1, rot_2, rot_3,"
        << "vel_x, vel_y, vel_z,"
        << "rot_vel_x, rot_vel_y, rot_vel_z,"
        << "acc_x, acc_y, acc_z,"
        << "rot_acc_x, rot_acc_y, rot_acc_z,"
        << "force_x, force_y, force_z,"
        << "torque_x, torque_y, torque_z\n";

    return dat;
}

void WriteData(std::ofstream& dat, ChSystemParallelSMC* msystem, const std::string& str, const int gcase) {
    for (int i = 0; i < msystem->Get_bodylist().size(); ++i) {
        const std::shared_ptr<ChBody> body = msystem->Get_bodylist().at(i);

        // Get the radius of the object
        ChCollisionModelParallel* pmodel = static_cast<ChCollisionModelParallel*>(body->GetCollisionModel().get());
        double radius = 0;
        switch (
            pmodel->mData[0].type) {  // I'm assuming that there is only one object per collision model. Fix this later.
            case chrono::collision::SPHERE:
                radius = pmodel->mData[0].B.x;
                break;
            case chrono::collision::BOX:
                radius = 1.0e8;
                break;
            case chrono::collision::CYLINDER:
                radius = pmodel->mData[0].B.x;
                break;
        }

        // Add all other object information to the data structure
        dat << str.c_str() << "," << gcase << "," << body->GetIdentifier() << "," << msystem->GetChTime() << ","
            << radius << "," << msystem->data_manager->host_data.ct_body_map[i] << "," << body->GetPos().x() << ","
            << body->GetPos().y() << "," << body->GetPos().z() << "," << body->GetRot().e0() << ","
            << body->GetRot().e1() << "," << body->GetRot().e2() << "," << body->GetRot().e3() << ","
            << body->GetPos_dt().x() << "," << body->GetPos_dt().y() << "," << body->GetPos_dt().z() << ","
            << body->GetWvel_par().x() << "," << body->GetWvel_par().y() << "," << body->GetWvel_par().z() << ","
            << body->GetPos_dtdt().x() << "," << body->GetPos_dtdt().y() << "," << body->GetPos_dtdt().z() << ","
            << body->GetWacc_par().x() << "," << body->GetWacc_par().y() << "," << body->GetWacc_par().z() << ","
            << msystem->GetBodyContactForce(body).x << "," << msystem->GetBodyContactForce(body).y << ","
            << msystem->GetBodyContactForce(body).z << "," << msystem->GetBodyContactTorque(body).x << ","
            << msystem->GetBodyContactTorque(body).y << "," << msystem->GetBodyContactTorque(body).z << "\n";
    }
}

int main(int argc, char* argv[]) {
    // Update accordingly. The rest of main should remain largely the same between projects.
    const std::string projname = "test_cohesion";

    if (CreateOutputPath(projname) != 0) {
        fprintf(stderr, "Error creating output data directory\n");
        return -1;
    }

    auto dat = CreateDataFile();

    // Print the sim set - up parameters to userlog
    GetLog() << "\nCopyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n";
    GetLog() << "\nTesting SMC Parallel cohesion behavior....\n";

    // Execute test for each force model
    std::vector<std::string> fmodels = {"hooke", "hertz", "plaincoulomb", "flores"};
    std::vector<ChVector<>> grav = {ChVector<>(0, -8, 0), ChVector<>(0, -10, 0)};

    for (int f = 0; f < fmodels.size(); ++f) {
        for (int g = 0; g < grav.size(); ++g) {
            // Create a shared material to be used by the all bodies
            float y_modulus = 2.0e5f;  /// Default 2e5
            float p_ratio = 0.3f;      /// Default 0.3f
            float s_frict = 0.3f;      /// Usually in 0.1 range, rarely above. Default 0.6f
            float k_frict = 0.3f;      /// Default 0.6f
            float roll_frict = 0.0f;   /// Usually around 1E-3
            float spin_frict = 0.0f;   /// Usually around 1E-3
            float cor_in = 0.0f;       /// Default 0.4f
            float ad = 10.0f;          /// Magnitude of the adhesion in the Constant adhesion model
            float adDMT = 0.0f;        /// Magnitude of the adhesion in the DMT adhesion model
            float adSPerko = 0.0f;     /// Magnitude of the adhesion in the SPerko adhesion model

            auto mat = chrono_types::make_shared<ChMaterialSurfaceSMC>();
            mat->SetYoungModulus(y_modulus);
            mat->SetPoissonRatio(p_ratio);
            mat->SetSfriction(s_frict);
            mat->SetKfriction(k_frict);
            mat->SetRollingFriction(roll_frict);
            mat->SetSpinningFriction(spin_frict);
            mat->SetRestitution(cor_in);
            mat->SetAdhesion(ad);
            mat->SetAdhesionMultDMT(adDMT);
            mat->SetAdhesionSPerko(adSPerko);

            // Create a parallel SMC system and set the system parameters
            double time_step = 1.0E-5;
            double out_step = 1.0E-2;
            double time_sim = 1.0;

            ChVector<> gravity(0, 0, 0);

            ChSystemParallelSMC msystem;
            SetSimParameters(&msystem, gravity, force_to_enum(fmodels[f]));

            // Add the sphere to the system
            double srad = 0.5;
            double smass = 1.0;
            ChVector<> spos(0, srad + 1e-2, 0);
            ChVector<> init_v(0, -0.1, 0);

            auto body1 = AddSphere(0, &msystem, mat, srad, smass, spos, init_v);
            auto body2 = AddSphere(1, &msystem, mat, srad, smass, spos * -1, init_v * -1);

            // Create the Irrlicht visualization.
#ifdef CHRONO_IRRLICHT
            bool vis = false;
            auto application = SetSimVis(&msystem, time_step, vis);
#endif

            // Print the sim set - up parameters to userlog once
            if (f == 0 && g == 0) {
                GetLog() << "\ntime_step, " << time_step << "\nout_step, " << out_step << "\ntotal_sim_time, "
                         << time_sim << "\nnum_sim_threads, " << CHOMPfunctions::GetNumThreads() << "\nadhesion_model, "
                         << static_cast<int>(msystem.GetSettings()->solver.adhesion_force_model)
                         << "\ntangential_displ_model, "
                         << static_cast<int>(msystem.GetSettings()->solver.tangential_displ_mode) << "\ntimestepper, "
                         << static_cast<int>(msystem.GetTimestepperType()) << "\ngravity_x, " << grav[0].x() << " "
                         << grav[1].x() << "\ngravity_y, " << grav[0].y() << " " << grav[1].y() << "\ngravity_z, "
                         << grav[0].z() << " " << grav[1].z() << "\nyoungs_modulus, " << y_modulus
                         << "\npoissons_ratio, " << p_ratio << "\nstatic_friction, " << s_frict
                         << "\nkinetic_friction, " << k_frict << "\nrolling_friction, " << roll_frict
                         << "\nspinning_friction, " << spin_frict << "\ncor, " << cor_in << "\nconstant_adhesion, "
                         << ad << "\nDMT_adhesion_multiplier, " << adDMT << "\nperko_adhesion_multiplier, " << adSPerko
                         << "\n";
            }
            GetLog() << "\nModel #" << f << " (" << fmodels[f].c_str() << ")  gravity setting #" << g << "\n";

            // Set the soft-real-time cycle parameters
            double time = 0.0;
            double out_time = 0.0;

            // Let the block settle of the plate before giving it a push
            while (time < time_sim) {
                while (time == 0 || time < out_time) {
                    msystem.DoStepDynamics(time_step);
                    time += time_step;
                }

                out_time = time - time_step + out_step;

                bool KEthresh = CalcAverageKE(&msystem, 1.0E-9);
                if (KEthresh) {
                    GetLog() << "[settling] KE exceeds threshold at t = " << time << "\n";
                    break;
                }
            }

            // Write the initial state date, fix the top sphere, and turn on gravity
            WriteData(dat, &msystem, fmodels[f].c_str(), g);
            body1->SetBodyFixed(true);

            gravity = grav[g];
            msystem.Set_G_acc(gravity);

            // Re-set the clock
            time_sim = time + 0.4;

            while (time < time_sim) {
#ifdef CHRONO_IRRLICHT
                if (application != NULL) {
                    application->BeginScene(true, true, SColor(255, 255, 255, 255));
                    application->DrawAll();
                }
#endif

                while (time == 0 || time < out_time) {
                    msystem.DoStepDynamics(time_step);
                    time += time_step;
                }

                out_time = time - time_step + out_step;

#ifdef CHRONO_IRRLICHT
                if (application != NULL)
                    application->EndScene();
#endif

                WriteData(dat, &msystem, fmodels[f].c_str(), g);
            }

            // Delete Visualization and check the results
#ifdef CHRONO_IRRLICHT
            if (application != NULL)
                delete application;
#endif
        }

        // Check results. The spheres should remain together when F_gravity < F_adhesion
        // but separate when F_gravity >= F_adhesion
    }

    dat.close();

    return 0;
}