
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
// Authors: Hammad Mazhar
// =============================================================================
//
// ChronoParallel test program for rolling friction
//
// The global reference frame has Y up.
//
// =============================================================================

#include "chrono/ChConfig.h"
#include "chrono/utils/ChUtilsCreators.h"

#include "chrono_parallel/physics/ChSystemParallel.h"
#include "chrono_parallel/solver/ChIterativeSolverParallel.h"

#ifdef CHRONO_OPENGL
#include "chrono_opengl/ChOpenGLWindow.h"
#endif

using namespace chrono;

// --------------------------------------------------------------------------

int main(int argc, char** argv) {
    // ----------------
    // Parameters
    // ----------------
    double radius = 0.5;
    double density = 1000;
    double mass = density * (4.0 / 3.0) * CH_C_PI * pow(radius, 3);
    double inertia = (2.0 / 5.0) * mass * pow(radius, 2);
    double initial_angspeed = 10;
    double initial_linspeed = initial_angspeed * radius;

    float sliding_friction = 0.1f;
    float rolling_friction = 0.1f;

    double time_step = 1e-3;

    double tolerance = 0;
    double contact_recovery_speed = 1e8;
    double collision_envelope = .05 * radius;

    uint max_iteration_normal = 0;
    uint max_iteration_sliding = 0;
    uint max_iteration_spinning = 100;
    uint max_iteration_bilateral = 0;
    
    // ------------------------
    // Create the parallel system
    // --------------------------

    ChSystemParallelDVI system;
    system.Set_G_acc(ChVector<>(0, -10, 0));

    // Set number of threads
    system.SetParallelThreadNumber(1);
    CHOMPfunctions::SetNumThreads(1);

    // Set solver settings
    system.ChangeSolverType(SolverType::APGD);

    system.GetSettings()->perform_thread_tuning = false;

    system.GetSettings()->solver.solver_mode = SolverMode::SPINNING;
    system.GetSettings()->solver.max_iteration_normal = max_iteration_normal;
    system.GetSettings()->solver.max_iteration_sliding = max_iteration_sliding;
    system.GetSettings()->solver.max_iteration_spinning = max_iteration_spinning;
    system.GetSettings()->solver.max_iteration_bilateral = max_iteration_bilateral;
    system.GetSettings()->solver.alpha = 0;
    system.GetSettings()->solver.contact_recovery_speed = contact_recovery_speed;
    system.GetSettings()->solver.use_full_inertia_tensor = false;
    system.GetSettings()->solver.tolerance = tolerance;

    system.GetSettings()->collision.collision_envelope = collision_envelope;
    system.GetSettings()->collision.narrowphase_algorithm = NarrowPhaseType::NARROWPHASE_R;
    system.GetSettings()->collision.bins_per_axis = vec3(10, 10, 10);

    // ----------
    // Add bodies
    // ----------

	auto container = std::shared_ptr<ChBody>(system.NewBody());
	system.Add(container);
	container->SetPos(ChVector<>(0, 0, 0));
	container->SetBodyFixed(true);
	container->SetIdentifier(-1);

	container->GetMaterialSurface()->SetFriction(sliding_friction);
	container->GetMaterialSurface()->SetRollingFriction(rolling_friction);

	container->SetCollide(true);
	container->GetCollisionModel()->ClearModel();
	utils::AddBoxGeometry(container.get(), ChVector<>(20, .5, 20), ChVector<>(0, -.5, 0));
	container->GetCollisionModel()->BuildModel();

    container->AddAsset(std::make_shared<ChColorAsset>(ChColor(0.4f, 0.4f, 0.2f)));

    auto ball = std::shared_ptr<ChBody>(system.NewBody());
	ChVector<> pos = ChVector<>(0, radius, 0);
	ChVector<> vel = ChVector<>(initial_linspeed, 0, 0);
	ChVector<> wvel = ChVector<>(0, 0, -initial_angspeed);
	ball->SetMass(mass);
	ball->SetPos(pos);
	ball->SetPos_dt(vel);
	ball->SetWvel_par(wvel);
	ball->SetInertiaXX(ChVector<>(inertia));

    ball->GetMaterialSurface()->SetFriction(sliding_friction);
    ball->GetMaterialSurface()->SetRollingFriction(rolling_friction);

    ball->SetCollide(true);
    ball->GetCollisionModel()->ClearModel();
	utils::AddSphereGeometry(ball.get(), radius);
	ball->GetCollisionModel()->BuildModel();

    ball->AddAsset(std::make_shared<ChColorAsset>(ChColor(0.2f, 0.3f, 0.4f)));

	system.AddBody(ball);

#ifdef CHRONO_OPENGL
    // -------------------------------
    // Create the visualization window
    // -------------------------------

    opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
    gl_window.Initialize(1280, 720, "Rolling test", &system);
    gl_window.SetCamera(ChVector<>(10, 10, 20), ChVector<>(0, 0, 0), ChVector<>(0, 1, 0), 0.05f);
    gl_window.SetRenderMode(opengl::WIREFRAME);
#endif

    // ---------------
    // Simulate system
    // ---------------

    double time_end = 20.0;
    double time_out = 2.5;
    bool output = false;

    while (system.GetChTime() < time_end) {
        system.DoStepDynamics(time_step);

        auto pos = ball->GetPos();
        printf("T: %f  Pos: %f %f %f\n", system.GetChTime(), pos.x(), pos.y(), pos.z());

        //if (!output && system.GetChTime() >= time_out) {
        //    for (int i = 1; i <= 10; i++) {
        //        auto pos = system.Get_bodylist()->at(i)->GetPos();
        //        std::cout << pos.x() << std::endl;
        //    }
        //    output = true;
        //}
		
#ifdef CHRONO_OPENGL
        opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
        if (gl_window.Active()) {
            gl_window.Render();
        } else {
            return 1;
        }
#endif
    }

    return 0;
}