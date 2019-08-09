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
// ChronoParallel test program for changing gravity direction during simulation.
// Uses non-smooth method for frictional contact.
//
// The global reference frame has Z up.
//
// If available, OpenGL is used for run-time rendering. Otherwise, the
// simulation is carried out for a pre-defined duration and output files are
// generated for post-processing with POV-Ray.
// =============================================================================

#include <cstdio>
#include <vector>
#include <cmath>

#include "chrono_parallel/physics/ChSystemParallel.h"

#include "chrono/ChConfig.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#ifdef CHRONO_OPENGL
#include "chrono_opengl/ChOpenGLWindow.h"
#endif

using namespace chrono;
using namespace chrono::collision;

// Tilt angle (about global Y axis) of the container.
double tilt_angle = 0 * CH_C_PI / 20;

// Number of balls: (2 * count_X + 1) * (2 * count_Y + 1)
int count_X = 2;
int count_Y = 2;

// -----------------------------------------------------------------------------
// Create a bin consisting of five boxes attached to the ground.
// -----------------------------------------------------------------------------
void AddContainer(ChSystemParallelNSC* sys) {
  // IDs for the two bodies
  int binId = -200;

  // Create a common material
  auto mat = chrono_types::make_shared<ChMaterialSurfaceNSC>();
  mat->SetFriction(0.4f);

  // Create the containing bin (4 x 4 x 1)
  auto bin = chrono_types::make_shared<ChBody>(chrono_types::make_shared<ChCollisionModelParallel>());
  bin->SetMaterialSurface(mat);
  bin->SetIdentifier(binId);
  bin->SetMass(1);
  bin->SetPos(ChVector<>(0, 0, 0));
  bin->SetRot(Q_from_AngY(tilt_angle));
  bin->SetCollide(true);
  bin->SetBodyFixed(true);

  ChVector<> hdim(2, 2, 0.5);
  double hthick = 0.1;

  bin->GetCollisionModel()->ClearModel();
  utils::AddBoxGeometry(bin.get(), ChVector<>(hdim.x(), hdim.y(), hthick), ChVector<>(0, 0, -hthick));
  utils::AddBoxGeometry(bin.get(), ChVector<>(hthick, hdim.y(), hdim.z()), ChVector<>(-hdim.x() - hthick, 0, hdim.z()));
  utils::AddBoxGeometry(bin.get(), ChVector<>(hthick, hdim.y(), hdim.z()), ChVector<>(hdim.x() + hthick, 0, hdim.z()));
  utils::AddBoxGeometry(bin.get(), ChVector<>(hdim.x(), hthick, hdim.z()), ChVector<>(0, -hdim.y() - hthick, hdim.z()));
  utils::AddBoxGeometry(bin.get(), ChVector<>(hdim.x(), hthick, hdim.z()), ChVector<>(0, hdim.y() + hthick, hdim.z()));
  bin->GetCollisionModel()->BuildModel();

  sys->AddBody(bin);
}

// -----------------------------------------------------------------------------
// Create the falling spherical objects in a uniform rectangular grid.
// -----------------------------------------------------------------------------
void AddFallingBalls(ChSystemParallel* sys) {
  // Common material
  auto ballMat = chrono_types::make_shared<ChMaterialSurfaceNSC>();
  ballMat->SetFriction(0.4f);

  // Create the falling balls
  int ballId = 0;
  double mass = 1;
  double radius = 0.15;
  ChVector<> inertia = (2.0 / 5.0) * mass * radius * radius * ChVector<>(1, 1, 1);

  for (int ix = -count_X; ix <= count_X; ix++) {
    for (int iy = -count_Y; iy <= count_Y; iy++) {
      ChVector<> pos(0.4 * ix, 0.4 * iy, 1);

      auto ball = chrono_types::make_shared<ChBody>(chrono_types::make_shared<ChCollisionModelParallel>());
      ball->SetMaterialSurface(ballMat);

      ball->SetIdentifier(ballId++);
      ball->SetMass(mass);
      ball->SetInertiaXX(inertia);
      ball->SetPos(pos);
      ball->SetRot(ChQuaternion<>(1, 0, 0, 0));
      ball->SetBodyFixed(false);
      ball->SetCollide(true);

      ball->GetCollisionModel()->ClearModel();
      utils::AddSphereGeometry(ball.get(), radius);
      ball->GetCollisionModel()->BuildModel();

      sys->AddBody(ball);
    }
  }
}

// -----------------------------------------------------------------------------
// Create the system, specify simulation parameters, and run simulation loop.
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  int threads = 8;

  // Simulation parameters
  // ---------------------

  ChVector<> gravity(0, 0, -10);

  uint max_iteration = 300;
  real tolerance = 1e-3;

  // Create system
  // -------------

  ChSystemParallelNSC msystem;

  // Set number of threads.
  int max_threads = CHOMPfunctions::GetNumProcs();
  if (threads > max_threads)
    threads = max_threads;
  msystem.SetParallelThreadNumber(threads);
  CHOMPfunctions::SetNumThreads(threads);

  // Set gravitational acceleration
  msystem.Set_G_acc(gravity);

  // Set solver parameters
  msystem.GetSettings()->solver.solver_mode = SolverMode::SLIDING;
  msystem.GetSettings()->solver.max_iteration_normal = max_iteration / 3;
  msystem.GetSettings()->solver.max_iteration_sliding = max_iteration / 3;
  msystem.GetSettings()->solver.max_iteration_spinning = 0;
  msystem.GetSettings()->solver.max_iteration_bilateral = max_iteration / 3;
  msystem.GetSettings()->solver.tolerance = tolerance;
  msystem.GetSettings()->solver.alpha = 0;
  msystem.GetSettings()->solver.contact_recovery_speed = 10000;
  msystem.ChangeSolverType(SolverType::APGDREF);
  msystem.GetSettings()->collision.narrowphase_algorithm = NarrowPhaseType::NARROWPHASE_HYBRID_MPR;

  msystem.GetSettings()->collision.collision_envelope = 0.01;
  msystem.GetSettings()->collision.bins_per_axis = vec3(10, 10, 10);

  // Create the fixed and moving bodies
  // ----------------------------------

  AddContainer(&msystem);
  AddFallingBalls(&msystem);

// Perform the simulation
// ----------------------

#ifdef CHRONO_OPENGL
  opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
  gl_window.Initialize(1280, 720, "ballsDVI", &msystem);
  gl_window.SetCamera(ChVector<>(0, -5, 0), ChVector<>(0, 0, 0), ChVector<>(0, 0, 1));
  gl_window.SetRenderMode(opengl::WIREFRAME);
#endif

  ChMatrix33<> rot(CH_C_PI / 6, ChVector<>(0, 1, 0));
  ChVector<> gravity_1 = rot * gravity;
  ChVector<> gravity_2(0, -10, 0);

  double time_step = 1e-3;
  double time_end = 6;
  double time_1 = 1;
  double time_2 = 4;
  double time = 0;

  while (time < time_end) {
      if (time > time_1) {
          msystem.Set_G_acc(gravity_1);
      }
      if (time > time_2) {
          msystem.Set_G_acc(gravity_2);
      }

      msystem.DoStepDynamics(time_step);

#ifdef CHRONO_OPENGL
      if (gl_window.Active()) {
          gl_window.Render();
      } else {
          break;
      }
#endif

    time += time_step;
  }


  return 0;
}
