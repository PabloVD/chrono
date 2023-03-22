// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2022 projectchrono.org
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
// Polaris wheeled vehicle on custom external terrain
//
// =============================================================================

#include <cstdio>
#include <string>
#include <array>

#include "chrono/physics/ChSystemNSC.h"
#include "chrono/physics/ChSystemSMC.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono_vehicle/ChVehicleModelData.h"
#include "chrono_vehicle/ChWorldFrame.h"
#include "chrono_vehicle/utils/ChUtilsJSON.h"
#include "chrono_vehicle/wheeled_vehicle/vehicle/WheeledVehicle.h"

#include "chrono_vehicle/wheeled_vehicle/ChWheeledVehicleVisualSystemIrrlicht.h"

using namespace chrono;
using namespace chrono::vehicle;

using std::cout;
using std::cerr;
using std::cin;
using std::endl;

// -----------------------------------------------------------------------------

class CustomTerrain : public ChTerrain {
  public:
    CustomTerrain(WheeledVehicle& vehicle);
    void Synchronize();
    virtual void Advance(double step) override;

  private:
    const WheeledVehicle& m_vehicle;
    std::array<std::shared_ptr<ChWheel>, 4> m_wheels;
    std::array<TerrainForce, 4> m_tire_forces;
};

CustomTerrain::CustomTerrain(WheeledVehicle& vehicle) : m_vehicle(vehicle) {
    m_wheels[0] = vehicle.GetWheel(0, LEFT);
    m_wheels[1] = vehicle.GetWheel(0, RIGHT);
    m_wheels[2] = vehicle.GetWheel(1, LEFT);
    m_wheels[3] = vehicle.GetWheel(1, RIGHT);

    // Create a ground body (only to carry some visualization assets)
    auto ground = chrono_types::make_shared<ChBody>();
    ground->SetBodyFixed(true);
    ground->SetCollide(false);

    auto box = chrono_types::make_shared<ChBoxShape>();
    box->GetBoxGeometry().Size = ChVector<>(10, 10, 0.5);
    // box->SetColor(ChColor(0.2f, 0.3f, 1.0f));
    box->SetTexture(GetChronoDataFile("textures/concrete.jpg"), 10, 10);
    ground->AddVisualShape(box, ChFrame<>(ChVector<>(0, 0, -0.5), QUNIT));

    // vehicle.GetSystem()->AddBody(ground);
}

void CustomTerrain::Synchronize() {
    // 1. Calculate tire forces for each vehicle wheel.
    //    These forces are assumed expressed in the absolute frame and applied to the spindle center.

    // The example here only calculates a tire normal load.
    // - Consider the tire is a disc and perform a simple collision detection with the terrain, assumed to be a
    //   horizontal plane at 0 height. This is a copy of ChTire::DiscTerrainCollision1pt().
    // - Use a simple spring-damper model to calculate vertical load based on penetration depth.
    // - Add some damping in lateral direction to prevent drift (sliding).

    double terrain_height = 0;
    ChVector<> terrain_normal(0, 0, 1);

    double tire_stiffness = 175000;
    double tire_damping = 350;

    for (int i = 0; i < 4; i++) {
        // Tire radius
        double disc_radius = m_wheels[i]->GetTire()->GetRadius();

        // Wheel state
        WheelState wheel_state = m_wheels[i]->GetState();
        ChVector<> disc_center = wheel_state.pos;

        // Extract the wheel normal (expressed in global frame)
        ChMatrix33<> A(wheel_state.rot);
        ChVector<> disc_normal = A.Get_A_Yaxis();

        // Find the location of the lowest point on the wheel disc in the direction of the world vertical.
        ChVector<> wheel_forward = Vcross(disc_normal, ChWorldFrame::Vertical());
        wheel_forward.Normalize();
        ChVector<> wheel_bottom_location = disc_center + disc_radius * Vcross(disc_normal, wheel_forward);

        // No contact if the disc center is below the terrain.
        double disc_height = ChWorldFrame::Height(disc_center);
        if (disc_height <= terrain_height)
            continue;

        // Calculate the contact depth at this point.
        double bottom_height = ChWorldFrame::Height(wheel_bottom_location);
        double depth = (terrain_height - bottom_height) * Vdot(ChWorldFrame::Vertical(), terrain_normal);

        // Based on the sampled normal we now do a first order approximation of where the contact point
        // would be. We also will estimate the contact depth at that point.
        ChVector<> wheel_forward_normal = Vcross(disc_normal, terrain_normal);

        // There is no contact if the disc is (almost) horizontal, so bail out in that case.
        double sinTilt2 = wheel_forward_normal.Length2();
        if (sinTilt2 < 1e-3)
            continue;

        wheel_forward_normal.Normalize();

        // Now re-calculate the depth.
        depth = disc_radius - ((disc_radius - depth) * Vdot(wheel_forward, wheel_forward_normal));

        // At this point we should check if our wheel still touches the ground and bail out if it does not.
        if (depth <= 0.0)
            continue;

        // And we re-calculate the contact point.
        wheel_bottom_location = disc_center + (disc_radius - depth) * Vcross(disc_normal, wheel_forward_normal);

        // Create the tire frame
        ChVector<> longitudinal = Vcross(disc_normal, terrain_normal);
        longitudinal.Normalize();
        ChVector<> lateral = Vcross(terrain_normal, longitudinal);
        ChMatrix33<> rot;
        rot.Set_A_axis(longitudinal, lateral, terrain_normal);
        ChFrame<> frame(wheel_bottom_location, rot);

        // Get wheel velocity expressed in the tire frame
        ChVector<> wheel_vel = frame.TransformDirectionParentToLocal(wheel_state.lin_vel);

        // Calculate the tire vertical load assuming a spring-damper model
        double Fn = tire_stiffness * depth - tire_damping * wheel_vel.z();

        // Add some damping force in the lateral direction to prevent drift
        double Fy = -tire_damping * wheel_vel.y();

        // Tire force and moment in tire frame
        ChVector<> tire_F(0, Fy, Fn);
        ChVector<> tire_M(0, 0, 0);

        // Load the tire force structure (all expressed in absolute frame)
        m_tire_forces[i].force = frame.TransformDirectionLocalToParent(tire_F);
        m_tire_forces[i].moment = frame.TransformDirectionLocalToParent(tire_M);
        m_tire_forces[i].point = wheel_bottom_location;
    }

    // 2. Add tire forces as external forces to spindle bodies

    for (int i = 0; i < 4; i++) {
        m_wheels[i]->GetSpindle()->Accumulate_force(m_tire_forces[i].force, m_tire_forces[i].point, false);
        m_wheels[i]->GetSpindle()->Accumulate_torque(m_tire_forces[i].moment, false);
    }
}

void CustomTerrain::Advance(double step) {}

// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    // Create the Chrono system
    ChSystemNSC sys;
    sys.Set_G_acc(ChVector<>(0, 0, -9.81));

    // Create vehicle
    ChCoordsys<> init_pos(ChVector<>(0, 0, 0.5), QUNIT);

    std::string vehicle_json = "Polaris/Polaris.json";
    std::string powertrain_json = "Polaris/Polaris_SimpleMapPowertrain.json";
    std::string tire_json = "Polaris/Polaris_RigidTire.json";

    // Create and initialize the vehicle
    WheeledVehicle vehicle(&sys, vehicle::GetDataFile(vehicle_json));
    vehicle.Initialize(init_pos);
    vehicle.GetChassis()->SetFixed(false);
    vehicle.SetChassisVisualizationType(VisualizationType::MESH);
    vehicle.SetSuspensionVisualizationType(VisualizationType::PRIMITIVES);
    vehicle.SetSteeringVisualizationType(VisualizationType::PRIMITIVES);
    vehicle.SetWheelVisualizationType(VisualizationType::MESH);

    // Create and initialize the powertrain system
    auto powertrain = ReadPowertrainJSON(vehicle::GetDataFile(powertrain_json));
    vehicle.InitializePowertrain(powertrain);

    // Create and initialize the tires
    for (auto& axle : vehicle.GetAxles()) {
        for (auto& wheel : axle->GetWheels()) {
            auto tire = ReadTireJSON(vehicle::GetDataFile(tire_json));
            vehicle.InitializeTire(tire, wheel, VisualizationType::MESH);
        }
    }

    // Create terrain
    CustomTerrain terrain(vehicle);

    // Create Irrilicht visualization
    ChWheeledVehicleVisualSystemIrrlicht vis;
    vis.SetWindowTitle("Polaris - Custom terrain example");
    vis.SetChaseCamera(ChVector<>(0.0, 0.0, 1.75), 5.0, 0.5);
    vis.Initialize();
    vis.AddTypicalLights();
    vis.AddSkyBox();
    vis.AddLogo();
    vis.AttachVehicle(&vehicle);

    // Simulation loop
    DriverInputs driver_inputs = {0, 0, 0};

    double step_size = 1e-3;
    while (vis.Run()) {
        double time = sys.GetChTime();

        vis.BeginScene();
        vis.Render();
        vis.EndScene();

        // Synchronize subsystems
        vehicle.Synchronize(time, driver_inputs, terrain);
        terrain.Synchronize();

        // Advance system state
        vehicle.Advance(step_size);
        terrain.Advance(step_size);
        vis.Advance(step_size);
        sys.DoStepDynamics(step_size);
    }

    return 0;
}