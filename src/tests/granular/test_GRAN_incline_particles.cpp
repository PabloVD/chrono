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
// Author: Radu Serban
// =============================================================================
//
// Test 3-DOF particle simulation with Chrono::Parallel
//
// Contact uses non-smooth (DVI) formulation.
// The global reference frame has Z up.
// All units SI.
// =============================================================================

#include <iostream>
#include <vector>
#include <cmath>

#include "chrono/ChConfig.h"
#include "chrono/core/ChFileutils.h"
#include "chrono/core/ChStream.h"
#include "chrono/utils/ChUtilsGeometry.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono_parallel/physics/ChSystemParallel.h"
#include "chrono_parallel/solver/ChIterativeSolverParallel.h"
#include "chrono_parallel/physics/Ch3DOFContainer.h"

#ifdef CHRONO_OPENGL
#include "chrono_opengl/ChOpenGLWindow.h"
#endif

using namespace chrono;
using namespace chrono::collision;

using std::cout;
using std::endl;

// -----------------------------------------------------------------------------
// Specification of the terrain
// -----------------------------------------------------------------------------

// Use 3DOF particles?
bool use_particles = true;

// Container
double hdimX = 0.5;
double hdimY = 0.25;
double hdimZ = 0.5;
double hthick = 0.05;

bool rough = true;

// Number of layers of granular material
int num_layers = 4;

// Granular material properties
int radius_mm = 10;        // particle radius (mm)
double density = 2000;     // density (Kg/m3)
double friction = 0.5;     // coefficient of friction
double compliance = 1e-9;  // compliance
int coh_pressure_kpa = 1;  // cohesion pressure (kPa)

int slope_deg = 30;                          // terrain slope (degrees)
double slope = (CH_C_PI / 180) * slope_deg;  // terrain slope (radians)

// -----------------------------------------------------------------------------
// Timed events
// -----------------------------------------------------------------------------

// Total simulation duration.
double time_end = 1;

// Time when terrain is pitched (rotate gravity)
double time_pitch = 0.15;

// -----------------------------------------------------------------------------
// Simulation parameters
// -----------------------------------------------------------------------------

// Desired number of OpenMP threads (will be clamped to maximum available)
int threads = 20;
bool thread_tuning = false;

// Integration step size
double time_step = 1e-3;

// Solver settings
int max_iteration_bilateral = 100;
int max_iteration_normal = 0;
int max_iteration_sliding = 50;
int max_iteration_spinning = 0;

double tolerance = 1e-3;
double alpha = 0;
double contact_recovery_speed = 1000;

// Output
bool render = true;
bool output = false;
std::string out_dir = "../PARTICLES";

std::shared_ptr<ChParticleContainer> particle_container;

// =============================================================================
// Custom material composition laws
class CustomCompositionStrategy : public ChMaterialCompositionStrategy<real> {
  public:
    virtual real CombineFriction(real a1, real a2) const override { return std::max<real>(a1, a2); }
};

// =============================================================================
// Utility function to print to console a few important step statistics

static inline void TimingOutput(chrono::ChSystem* mSys, chrono::ChStreamOutAsciiFile* ofile = NULL) {
    double TIME = mSys->GetChTime();
    double STEP = mSys->GetTimerStep();
    double BROD = mSys->GetTimerCollisionBroad();
    double NARR = mSys->GetTimerCollisionNarrow();
    double SOLVER = mSys->GetTimerSolver();
    double UPDT = mSys->GetTimerUpdate();
    double RESID = 0;
    int REQ_ITS = 0;
    int BODS = mSys->GetNbodies();
    int CNTC = mSys->GetNcontacts();
    if (chrono::ChSystemParallel* parallel_sys = dynamic_cast<chrono::ChSystemParallel*>(mSys)) {
        RESID = std::static_pointer_cast<chrono::ChIterativeSolverParallel>(mSys->GetSolver())->GetResidual();
        REQ_ITS = std::static_pointer_cast<chrono::ChIterativeSolverParallel>(mSys->GetSolver())->GetTotalIterations();
        BODS = parallel_sys->GetNbodies();
        CNTC = parallel_sys->GetNcontacts();
    }

    if (ofile) {
        char buf[200];
        sprintf(buf, "%8.5f  %7.4f  %7.4f  %7.4f  %7.4f  %7.4f  %7d  %7d  %7d  %7.4f\n", TIME, STEP, BROD, NARR, SOLVER,
                UPDT, BODS, CNTC, REQ_ITS, RESID);
        *ofile << buf;
    }

    printf("   %8.5f | %7.4f | %7.4f | %7.4f | %7.4f | %7.4f | %7d | %7d | %7d | %7.4f\n", TIME, STEP, BROD, NARR,
           SOLVER, UPDT, BODS, CNTC, REQ_ITS, RESID);
}

// =============================================================================

double CreateContainer(ChSystem* system,  // containing system
                       double mu,         // coefficient of friction
                       double coh,        // cohesion (constant impulse)
                       double radius      // radius of roughness spheres (if positive)
                       ) {
    bool visible_walls = false;

    auto material = std::make_shared<ChMaterialSurfaceNSC>();
    material->SetFriction((float)mu);
    material->SetCohesion((float)coh);
    material->SetCompliance((float)compliance);

    auto ground = std::make_shared<ChBody>(std::make_shared<ChCollisionModelParallel>());
    ground->SetIdentifier(-1);
    ground->SetMass(1000);
    ground->SetBodyFixed(true);
    ground->SetCollide(true);

    ground->SetMaterialSurface(material);

    ground->GetCollisionModel()->ClearModel();

    // Bottom box
    utils::AddBoxGeometry(ground.get(), ChVector<>(hdimX, hdimY, hthick), ChVector<>(0, 0, -hthick),
                          ChQuaternion<>(1, 0, 0, 0), true);
    // Left box
    utils::AddBoxGeometry(ground.get(), ChVector<>(hdimX, hthick, hdimZ + hthick),
                          ChVector<>(0, hdimY + hthick, hdimZ - hthick), ChQuaternion<>(1, 0, 0, 0), visible_walls);
    // Right box
    utils::AddBoxGeometry(ground.get(), ChVector<>(hdimX, hthick, hdimZ + hthick),
                          ChVector<>(0, -hdimY - hthick, hdimZ - hthick), ChQuaternion<>(1, 0, 0, 0), visible_walls);

    // Front box
    utils::AddBoxGeometry(ground.get(), ChVector<>(hthick, hdimY, hdimZ + hthick),
                          ChVector<>(hdimX + hthick, 0, hdimZ - hthick), ChQuaternion<>(1, 0, 0, 0), visible_walls);
    // Rear box
    utils::AddBoxGeometry(ground.get(), ChVector<>(hthick, hdimY, hdimZ + hthick),
                          ChVector<>(-hdimX - hthick, 0, hdimZ - hthick), ChQuaternion<>(1, 0, 0, 0), visible_walls);

    // If a positive radius was provided, create a "rough" surface
    if (radius > 0) {
        double d = 4 * radius;
        int nx = (int)std::floor(hdimX / d);
        int ny = (int)std::floor(hdimY / d);
        for (int ix = -nx; ix <= nx; ix++) {
            for (int iy = -ny; iy <= ny; iy++) {
                utils::AddSphereGeometry(ground.get(), radius, ChVector<>(ix * d, iy * d, radius));
            }
        }
    }

    ground->GetCollisionModel()->BuildModel();

    system->AddBody(ground);

    return (radius > 0) ? 2 * radius : 0;
}

// =============================================================================

unsigned int CreateParticles(ChSystemParallelNSC* system,  // containing system
                             double radius,                // particle radius
                             double rho,                   // particle density
                             double mu,                    // coefficient of friction
                             double coh,                   // cohesion (constant impulse)
                             double top_height             // top height of rigid container
                             ) {
    unsigned int num_particles = 0;

    if (use_particles) {
        particle_container = std::make_shared<ChParticleContainer>();
        system->Add3DOFContainer(particle_container);

        particle_container->kernel_radius = 2 * radius;
        particle_container->mass = rho * (4 * CH_C_PI / 3) * std::pow(radius, 3);

        particle_container->contact_mu = mu;
        particle_container->contact_cohesion = coh;
        particle_container->contact_compliance = compliance;

        particle_container->mu = mu;
        particle_container->cohesion = coh;
        particle_container->compliance = compliance;

        particle_container->alpha = alpha;

        particle_container->contact_recovery_speed = contact_recovery_speed;
        particle_container->collision_envelope = 0.1 * radius;

        std::vector<real3> pos;
        std::vector<real3> vel;

        double r = 1.01 * radius;
        ChVector<> hdims(hdimX - r, hdimY - r, 0);
        ChVector<> center(0, 0, top_height + 2 * r);

        utils::PDSampler<> sampler(2 * r);
        for (int il = 0; il < num_layers; il++) {
            utils::Generator::PointVector points = sampler.SampleBox(center, hdims);
            center.z() += 2 * r;
            for (int i = 0; i < points.size(); i++) {
                pos.push_back(real3(points[i].x(), points[i].y(), points[i].z()));
                vel.push_back(real3(0, 0, 0));
            }
        }

        particle_container->UpdatePosition(0);
        particle_container->AddBodies(pos, vel);

        num_particles = (unsigned int)pos.size();
    } else {
        // Create a material
        auto mat_g = std::make_shared<ChMaterialSurfaceNSC>();
        mat_g->SetFriction((float)mu);
        mat_g->SetCohesion((float)coh);
        mat_g->SetCompliance(1e-9f);

        // Create a particle generator and a mixture entirely made out of spheres
        utils::Generator gen(system);
        std::shared_ptr<utils::MixtureIngredient> m1 = gen.AddMixtureIngredient(utils::SPHERE, 1.0);
        m1->setDefaultMaterial(mat_g);
        m1->setDefaultDensity(rho);
        m1->setDefaultSize(radius);

        // Set starting value for body identifiers
        gen.setBodyIdentifier(1);

        // Create particles in layers until reaching the desired number of particles
        double r = 1.01 * radius;
        ChVector<> hdims(hdimX - r, hdimY - r, 0);
        ChVector<> center(0, 0, top_height + 2 * r);

        for (int il = 0; il < num_layers; il++) {
            gen.createObjectsBox(utils::POISSON_DISK, 2 * r, center, hdims);
            center.z() += 2 * r;
        }

        num_particles = gen.getTotalNumBodies();
    }

    return num_particles;
}

// =============================================================================

ChVector<> CalculateCOM(ChSystem* system) {
    ChVector<> com(0, 0, 0);
    unsigned int count = 0;

    if (use_particles) {
        count = particle_container->GetNumParticles();
        for (uint i = 0; i < count; i++) {
            real3 pos = particle_container->GetPos(i);
            com += ChVector<>(pos.x, pos.y, pos.z);
        }
    } else {
        for (auto body : system->Get_bodylist()) {
            if (body->GetIdentifier()) {
                com += body->GetPos();
                count++;
            }
        }
    }

    return com / count;
}

// =============================================================================

int main(int argc, char* argv[]) {
    double radius = radius_mm / 1.0e3;                   // particle radius (m)
    double area = CH_C_PI * radius * radius;             // cross-section area (m2)
    double coh_force = area * (coh_pressure_kpa * 1e3);  // cohesion force (N)
    double cohesion = coh_force * time_step;             // cohesion impulse (Ns)

    // --------------------------
    // Create output directories
    // --------------------------

    std::ofstream outf;

    if (output) {
        if (ChFileutils::MakeDirectory(out_dir.c_str()) < 0) {
            cout << "Error creating directory " << out_dir << endl;
            return 1;
        }

        // Open the output file stream
        std::string fname = out_dir + (use_particles ? "/resultsP_" : "/resultsB_");
        fname += std::to_string(slope_deg) + "_" + std::to_string(radius_mm) + "_" + std::to_string(friction) + "_" +
                 std::to_string(coh_pressure_kpa) + ".dat";
        outf.open(fname, std::ios::out);
        outf.precision(7);
        outf << std::scientific;
    }

    // Delimiter in output file
    std::string del("  ");

    // -------------
    // Create system
    // -------------
    ChVector<> gravity(0, 0, -9.81);
    ChVector<> gravityR = ChMatrix33<>(slope, ChVector<>(0, 1, 0)) * gravity;

    ChSystemParallelNSC* system = new ChSystemParallelNSC();
    system->Set_G_acc(gravity);

    // Test using a custom material property composition strategy
    ////std::unique_ptr<CustomCompositionStrategy> strategy(new CustomCompositionStrategy);
    ////system->SetMaterialCompositionStrategy(std::move(strategy));

    // ----------------------
    // Set number of threads.
    // ----------------------
    int max_threads = omp_get_num_procs();
    if (threads > max_threads)
        threads = max_threads;
    system->SetParallelThreadNumber(threads);
    omp_set_num_threads(threads);
    cout << "Using " << threads << " threads" << endl;

    system->GetSettings()->perform_thread_tuning = thread_tuning;

    // ---------------------
    // Edit system settings
    // ---------------------

    system->GetSettings()->solver.tolerance = tolerance;
    system->GetSettings()->solver.solver_mode = SolverMode::SLIDING;
    system->GetSettings()->solver.max_iteration_normal = max_iteration_normal;
    system->GetSettings()->solver.max_iteration_sliding = max_iteration_sliding;
    system->GetSettings()->solver.max_iteration_spinning = max_iteration_spinning;
    system->GetSettings()->solver.max_iteration_bilateral = max_iteration_bilateral;
    system->GetSettings()->solver.compute_N = false;
    system->GetSettings()->solver.alpha = alpha;
    system->GetSettings()->solver.cache_step_length = true;
    system->GetSettings()->solver.use_full_inertia_tensor = false;
    system->GetSettings()->solver.contact_recovery_speed = contact_recovery_speed;
    system->GetSettings()->solver.bilateral_clamp_speed = 1e8;
    system->GetSettings()->min_threads = threads;
    system->ChangeSolverType(SolverType::BB);
    ////system->SetLoggingLevel(LoggingLevel::LOG_INFO);
    ////system->SetLoggingLevel(LoggingLevel::LOG_TRACE);

    system->GetSettings()->collision.collision_envelope = 0.1 * radius;
    system->GetSettings()->collision.narrowphase_algorithm = NarrowPhaseType::NARROWPHASE_HYBRID_MPR;
    system->GetSettings()->collision.bins_per_axis = vec3(100, 30, 2);
    system->GetSettings()->collision.fixed_bins = true;

    // Specify active box.
    system->GetSettings()->collision.use_aabb_active = false;
    system->GetSettings()->collision.aabb_min = real3(-1.1 * hdimX, -1.1 * hdimY, 0);
    system->GetSettings()->collision.aabb_max = real3(+1.1 * hdimX, +1.1 * hdimY, 10 * hdimZ);

    // -------------------
    // Create the terrain.
    // -------------------
    cout << "Cohesion: " << coh_pressure_kpa << " " << coh_force << " " << cohesion << endl;

    double top_height = CreateContainer(system, friction, cohesion, (rough ? radius : -1));
    unsigned int actual_num_particles = CreateParticles(system, radius, density, friction, cohesion, top_height);

    cout << "Created " << actual_num_particles << " particles." << endl;

#ifdef CHRONO_OPENGL
    // Initialize OpenGL
    if (render) {
        opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
        gl_window.Initialize(1280, 720, "Particles on incline", system);
        gl_window.SetCamera(ChVector<>(0, -hdimY - 1, 0), ChVector<>(0, 0, 0), ChVector<>(0, 0, 1), float(hdimY / 10));
        gl_window.SetRenderMode(opengl::WIREFRAME);
    }
#endif

    // ---------------
    // Simulation loop
    // ---------------

    double time = 0;
    int sim_frame = 0;
    double exec_time = 0;

    bool is_pitched = false;

    while (time < time_end) {
        // Rotate gravity vector
        if (!is_pitched && time > time_pitch) {
            cout << time << "    Pitch: " << gravityR.x() << " " << gravityR.y() << " " << gravityR.z() << endl;
            system->Set_G_acc(gravityR);
            is_pitched = true;
        }

        // Advance system state
        system->DoStepDynamics(time_step);

        if (output) {
            ChVector<> com = CalculateCOM(system);
            outf << time << " " << com.x() << " " << com.y() << " " << com.z() << "\n";
        }

#ifdef CHRONO_OPENGL
        if (render) {
            opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
            if (gl_window.Active())
                gl_window.Render();
            else
                break;
        }
#endif

        // Display performance metrics
        TimingOutput(system);

        // Update counters.
        time += time_step;
        sim_frame++;
        exec_time += system->GetTimerStep();
    }

    return 0;
}
