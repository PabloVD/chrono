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
// Wrapper classes for modeling an entire GD250 vehicle assembly
// (including the vehicle itself, the powertrain, and the tires).
//
// =============================================================================

#include "chrono/ChConfig.h"

#include "chrono_vehicle/ChVehicleModelData.h"

#include "chrono_models/vehicle/g-wagon/gd250.h"

namespace chrono {
namespace vehicle {
namespace gwagon {

// -----------------------------------------------------------------------------
GD250::GD250()
    : m_system(nullptr),
      m_vehicle(nullptr),
      m_contactMethod(ChContactMethod::NSC),
      m_chassisCollisionType(CollisionType::NONE),
      m_fixed(false),
      m_brake_locking(false),
      m_brake_type(BrakeType::SIMPLE),
      m_tireType(TireModelType::RIGID),
      m_tire_step_size(-1),
      m_steeringType(SteeringTypeWV::PITMAN_ARM),
      m_initPos(ChCoordsys<>(ChVector<>(0, 0, 1), QUNIT)),
      m_initFwdVel(0),
      m_initOmega({0, 0, 0, 0}),
      m_kinematic_mode(true),
      m_low_range(false),
      m_apply_drag(false) {}

GD250::GD250(ChSystem* system)
    : m_system(system),
      m_vehicle(nullptr),
      m_contactMethod(ChContactMethod::NSC),
      m_chassisCollisionType(CollisionType::NONE),
      m_fixed(false),
      m_brake_locking(false),
      m_brake_type(BrakeType::SIMPLE),
      m_tireType(TireModelType::RIGID),
      m_tire_step_size(-1),
      m_steeringType(SteeringTypeWV::PITMAN_ARM),
      m_initPos(ChCoordsys<>(ChVector<>(0, 0, 1), QUNIT)),
      m_initFwdVel(0),
      m_initOmega({0, 0, 0, 0}),
      m_kinematic_mode(true),
      m_low_range(false),
      m_apply_drag(false) {}

GD250::~GD250() {
    delete m_vehicle;
}

// -----------------------------------------------------------------------------
void GD250::SetAerodynamicDrag(double Cd, double area, double air_density) {
    m_Cd = Cd;
    m_area = area;
    m_air_density = air_density;

    m_apply_drag = true;
}

// -----------------------------------------------------------------------------
void GD250::Initialize() {
    // Create and initialize the GD250 vehicle
    m_vehicle = m_system
                    ? new GD250_Vehicle(m_system, m_fixed, m_brake_type, m_steeringType, m_chassisCollisionType, m_kinematic_mode, m_low_range)
                    : new GD250_Vehicle(m_fixed, m_brake_type, m_steeringType, m_contactMethod, m_chassisCollisionType, m_kinematic_mode, m_low_range);

    m_vehicle->SetInitWheelAngVel(m_initOmega);
    m_vehicle->Initialize(m_initPos, m_initFwdVel);

    // If specified, enable aerodynamic drag
    if (m_apply_drag) {
        m_vehicle->GetChassis()->SetAerodynamicDrag(m_Cd, m_area, m_air_density);
    }

    // Create and initialize the powertrain system
    auto engine = chrono_types::make_shared<GD250_EngineSimpleMap>("Engine");
    auto transmission = chrono_types::make_shared<GD250_AutomaticTransmissionSimpleMap>("Transmission");
    auto powertrain = chrono_types::make_shared<ChPowertrainAssembly>(engine, transmission);
    m_vehicle->InitializePowertrain(powertrain);

    // Create the tires and set parameters depending on type.
    switch (m_tireType) {
        case TireModelType::RIGID:
        case TireModelType::RIGID_MESH: {
            bool use_mesh = (m_tireType == TireModelType::RIGID_MESH);

            auto tire_FL = chrono_types::make_shared<GD250_RigidTire>("FL", use_mesh);
            auto tire_FR = chrono_types::make_shared<GD250_RigidTire>("FR", use_mesh);
            auto tire_RL = chrono_types::make_shared<GD250_RigidTire>("RL", use_mesh);
            auto tire_RR = chrono_types::make_shared<GD250_RigidTire>("RR", use_mesh);

            m_vehicle->InitializeTire(tire_FL, m_vehicle->GetAxle(0)->m_wheels[LEFT], VisualizationType::NONE);
            m_vehicle->InitializeTire(tire_FR, m_vehicle->GetAxle(0)->m_wheels[RIGHT], VisualizationType::NONE);
            m_vehicle->InitializeTire(tire_RL, m_vehicle->GetAxle(1)->m_wheels[LEFT], VisualizationType::NONE);
            m_vehicle->InitializeTire(tire_RR, m_vehicle->GetAxle(1)->m_wheels[RIGHT], VisualizationType::NONE);

            m_tire_mass = tire_FL->GetMass();

            break;
        }

        case TireModelType::TMEASY: {
            auto tire_FL = chrono_types::make_shared<GD250_TMeasyTireFront>("FL");
            auto tire_FR = chrono_types::make_shared<GD250_TMeasyTireFront>("FR");
            auto tire_RL = chrono_types::make_shared<GD250_TMeasyTireFront>("RL");
            auto tire_RR = chrono_types::make_shared<GD250_TMeasyTireFront>("RR");
            tire_FL->SetCollisionType(ChTire::CollisionType::ENVELOPE);
            tire_FR->SetCollisionType(ChTire::CollisionType::ENVELOPE);
            tire_RL->SetCollisionType(ChTire::CollisionType::ENVELOPE);
            tire_RR->SetCollisionType(ChTire::CollisionType::ENVELOPE);
            m_vehicle->InitializeTire(tire_FL, m_vehicle->GetAxle(0)->m_wheels[LEFT], VisualizationType::MESH);
            m_vehicle->InitializeTire(tire_FR, m_vehicle->GetAxle(0)->m_wheels[RIGHT], VisualizationType::MESH);
            m_vehicle->InitializeTire(tire_RL, m_vehicle->GetAxle(1)->m_wheels[LEFT], VisualizationType::MESH);
            m_vehicle->InitializeTire(tire_RR, m_vehicle->GetAxle(1)->m_wheels[RIGHT], VisualizationType::MESH);

            m_tire_mass = tire_FL->GetMass();

            break;
        }

        case TireModelType::PAC02: {
            auto tire_FL = chrono_types::make_shared<GD250_Pac02Tire>("FL");
            auto tire_FR = chrono_types::make_shared<GD250_Pac02Tire>("FR");
            auto tire_RL = chrono_types::make_shared<GD250_Pac02Tire>("RL");
            auto tire_RR = chrono_types::make_shared<GD250_Pac02Tire>("RR");

            m_vehicle->InitializeTire(tire_FL, m_vehicle->GetAxle(0)->m_wheels[LEFT], VisualizationType::NONE);
            m_vehicle->InitializeTire(tire_FR, m_vehicle->GetAxle(0)->m_wheels[RIGHT], VisualizationType::NONE);
            m_vehicle->InitializeTire(tire_RL, m_vehicle->GetAxle(1)->m_wheels[LEFT], VisualizationType::NONE);
            m_vehicle->InitializeTire(tire_RR, m_vehicle->GetAxle(1)->m_wheels[RIGHT], VisualizationType::NONE);

            m_tire_mass = tire_FL->GetMass();

            break;
        }

        default:
            break;
    }

    for (auto& axle : m_vehicle->GetAxles()) {
        for (auto& wheel : axle->GetWheels()) {
            if (m_tire_step_size > 0)
                wheel->GetTire()->SetStepsize(m_tire_step_size);
        }
    }

    m_vehicle->EnableBrakeLocking(m_brake_locking);

    // Recalculate vehicle mass, to properly account for all subsystems
    m_vehicle->InitializeInertiaProperties();
}

// -----------------------------------------------------------------------------
void GD250::Synchronize(double time, const DriverInputs& driver_inputs, const ChTerrain& terrain) {
    m_vehicle->Synchronize(time, driver_inputs, terrain);
}
// -----------------------------------------------------------------------------
void GD250::Advance(double step) {
    m_vehicle->Advance(step);
}

}  // namespace gwagon
}  // end namespace vehicle
}  // end namespace chrono
