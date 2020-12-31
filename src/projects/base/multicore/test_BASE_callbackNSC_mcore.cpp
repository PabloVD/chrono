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
// Authors: Radu Serban
// =============================================================================
//
// Chrono demonstration of using contact callbacks for non-smooth contacts
// (complementarity-based).
//
// The global reference frame has Y up.
//
// =============================================================================

#include <cstdio>
#include <cmath>

#include "chrono/assets/ChColorAsset.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono_multicore/physics/ChSystemMulticore.h"

#include "chrono_opengl/ChOpenGLWindow.h"

using namespace chrono;

// -----------------------------------------------------------------------------
// Callback class for modifying composite material
// -----------------------------------------------------------------------------
class ContactMaterial : public ChContactContainer::AddContactCallback {
  public:
    virtual void OnAddContact(const collision::ChCollisionInfo& contactinfo,
                              ChMaterialComposite* const material) override {
        // Downcast to appropriate composite material type
        auto mat = static_cast<ChMaterialCompositeNSC* const>(material);

        // Set different friction for left/right halfs
        float friction = (contactinfo.vpA.z() > 0) ? 0.3f : 0.8f;
        mat->static_friction = friction;
        mat->sliding_friction = friction;
    }
};

int main(int argc, char* argv[]) {
    // ----------------
    // Parameters
    // ----------------

    float friction = 0.3f;

    double tolerance = 0;
    double contact_recovery_speed = 1e8;
    double collision_envelope = .001;

    uint max_iteration_normal = 0;
    uint max_iteration_sliding = 0;
    uint max_iteration_spinning = 100;
    uint max_iteration_bilateral = 0;

    // -----------------
    // Create the system
    // -----------------

    ChSystemMulticoreNSC system;
    system.Set_G_acc(ChVector<>(0, -10, 0));

    // Set number of threads
    system.SetNumThreads(1);

    // Set solver settings
    system.ChangeSolverType(SolverType::APGD);

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
    system.GetSettings()->collision.narrowphase_algorithm = NarrowPhaseType::NARROWPHASE_HYBRID_MPR;
    system.GetSettings()->collision.bins_per_axis = vec3(10, 10, 10);

    // Shared contact material
    auto contact_mat = chrono_types::make_shared<ChMaterialSurfaceNSC>();
    contact_mat->SetFriction(friction);

    // ----------
    // Add bodies
    // ----------

    auto container = std::shared_ptr<ChBody>(system.NewBody());
    system.Add(container);
    container->SetPos(ChVector<>(0, 0, 0));
    container->SetBodyFixed(true);
    container->SetIdentifier(-1);

    container->SetCollide(true);
    container->GetCollisionModel()->SetEnvelope(collision_envelope);
    container->GetCollisionModel()->ClearModel();
    utils::AddBoxGeometry(container.get(), contact_mat, ChVector<>(4, 0.5, 4), ChVector<>(0, -0.5, 0));
    container->GetCollisionModel()->BuildModel();

    container->AddAsset(chrono_types::make_shared<ChColorAsset>(ChColor(0.4f, 0.4f, 0.4f)));

    auto box1 = std::shared_ptr<ChBody>(system.NewBody());
    box1->SetMass(10);
    box1->SetInertiaXX(ChVector<>(1, 1, 1));
    box1->SetPos(ChVector<>(-1, 0.21, -1));
    box1->SetPos_dt(ChVector<>(5, 0, 0));

    box1->SetCollide(true);
    box1->GetCollisionModel()->SetEnvelope(collision_envelope);
    box1->GetCollisionModel()->ClearModel();
    utils::AddBoxGeometry(box1.get(), contact_mat, ChVector<>(0.4, 0.2, 0.1));
    box1->GetCollisionModel()->BuildModel();

    box1->AddAsset(chrono_types::make_shared<ChColorAsset>(ChColor(0.1f, 0.1f, 0.4f)));

    system.AddBody(box1);

    auto box2 = std::shared_ptr<ChBody>(system.NewBody());
    box2->SetMass(10);
    box2->SetInertiaXX(ChVector<>(1, 1, 1));
    box2->SetPos(ChVector<>(-1, 0.21, +1));
    box2->SetPos_dt(ChVector<>(5, 0, 0));

    box2->SetCollide(true);
    box2->GetCollisionModel()->SetEnvelope(collision_envelope);
    box2->GetCollisionModel()->ClearModel();
    utils::AddBoxGeometry(box2.get(), contact_mat, ChVector<>(0.4, 0.2, 0.1));
    box2->GetCollisionModel()->BuildModel();

    box2->AddAsset(chrono_types::make_shared<ChColorAsset>(ChColor(0.4f, 0.1f, 0.1f)));

    system.AddBody(box2);

    // -------------------------------
    // Create the visualization window
    // -------------------------------

    opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
    gl_window.Initialize(1280, 720, "Chrono::Multicore callback test", &system);
    gl_window.SetCamera(ChVector<>(6, 6, 10), ChVector<>(0, 0, 0), ChVector<>(0, 1, 0), 0.05f);
    gl_window.SetRenderMode(opengl::WIREFRAME);

    // ---------------
    // Simulate system
    // ---------------

    auto cmaterial = chrono_types::make_shared<ContactMaterial>();
    system.GetContactContainer()->RegisterAddContactCallback(cmaterial);

    while (gl_window.Active()) {
        gl_window.DoStepDynamics(1e-3);
        gl_window.Render();
        ////std::cout << std::fixed << std::setprecision(8) << box1->GetPos().x() << "   " << box2->GetPos().x() << std::endl;
        std::cout << box1->GetPos().x() - box2->GetPos().x() << std::endl;
    }

    return 0;
}
