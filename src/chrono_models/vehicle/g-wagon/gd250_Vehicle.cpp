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
// GD250 vehicle model.
//
// =============================================================================

#include "chrono/assets/ChSphereShape.h"
#include "chrono/assets/ChTriangleMeshShape.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono_vehicle/ChVehicleModelData.h"

#include "chrono_models/vehicle/g-wagon/gd250_Vehicle.h"
#include "chrono_models/vehicle/g-wagon/gd250_BrakeSimple.h"
#include "chrono_models/vehicle/g-wagon/gd250_BrakeShafts.h"
#include "chrono_models/vehicle/g-wagon/gd250_Chassis.h"
#include "chrono_models/vehicle/g-wagon/gd250_Driveline4WD.h"
#include "chrono_models/vehicle/g-wagon/gd250_GAxle.h"
#include "chrono_models/vehicle/g-wagon/gd250_GAxleSimple.h"
#include "chrono_models/vehicle/g-wagon/gd250_RotaryArm.h"
#include "chrono_models/vehicle/g-wagon/gd250_ToeBarGAxle.h"
#include "chrono_models/vehicle/g-wagon/gd250_ToeBarGAxleSimple.h"
#include "chrono_models/vehicle/g-wagon/gd250_Wheel.h"

namespace chrono {
namespace vehicle {
namespace gwagon {

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
GD250_Vehicle::GD250_Vehicle(const bool fixed,
                             BrakeType brake_type,
                             SteeringTypeWV steering_model,
                             ChContactMethod contact_method,
                             CollisionType chassis_collision_type,
                             bool kinematic_mode,
                             bool low_range)
    : ChWheeledVehicle("GD250", contact_method), m_omega({0, 0, 0, 0}), m_kinematic(kinematic_mode), m_low_range(low_range) {
    Create(fixed, brake_type, steering_model, chassis_collision_type);
}

GD250_Vehicle::GD250_Vehicle(ChSystem* system,
                             const bool fixed,
                             BrakeType brake_type,
                             SteeringTypeWV steering_model,
                             CollisionType chassis_collision_type,
                             bool kinematic_mode,
                             bool low_range)
    : ChWheeledVehicle("GD250", system), m_omega({0, 0, 0, 0}), m_kinematic(kinematic_mode), m_low_range(low_range) {
    Create(fixed, brake_type, steering_model, chassis_collision_type);
}

void GD250_Vehicle::Create(bool fixed,
                           BrakeType brake_type,
                           SteeringTypeWV steering_model,
                           CollisionType chassis_collision_type) {
    // Create the chassis subsystem
    m_chassis = chrono_types::make_shared<GD250_Chassis>("Chassis", fixed, chassis_collision_type);

    // Create the steering subsystem
    m_steerings.resize(1);
    m_steerings[0] = chrono_types::make_shared<GD250_RotaryArm>("Steering");

    // Create the axle subsystems
    m_axles.resize(2);
    m_axles[0] = chrono_types::make_shared<ChAxle>();
    m_axles[1] = chrono_types::make_shared<ChAxle>();

    if (m_kinematic) {
        GetLog() << "Kinematic Mode selected\n";
        m_axles[0]->m_suspension = chrono_types::make_shared<GD250_ToeBarGAxleSimple>("FrontSusp");
        m_axles[1]->m_suspension = chrono_types::make_shared<GD250_GAxleSimple>("RearSusp");
    } else {
        GetLog() << "Compliance Mode selected\n";
        m_axles[0]->m_suspension = chrono_types::make_shared<GD250_ToeBarGAxle>("FrontSusp");
        m_axles[1]->m_suspension = chrono_types::make_shared<GD250_GAxle>("RearSusp");
    }

    m_axles[0]->m_wheels.resize(2);
    m_axles[0]->m_wheels[0] = chrono_types::make_shared<GD250_Wheel>("Wheel_FL");
    m_axles[0]->m_wheels[1] = chrono_types::make_shared<GD250_Wheel>("Wheel_FR");
    m_axles[1]->m_wheels.resize(2);
    m_axles[1]->m_wheels[0] = chrono_types::make_shared<GD250_Wheel>("Wheel_RL");
    m_axles[1]->m_wheels[1] = chrono_types::make_shared<GD250_Wheel>("Wheel_RR");

    switch (brake_type) {
        case BrakeType::SIMPLE:
            m_axles[0]->m_brake_left = chrono_types::make_shared<GD250_BrakeSimpleFront>("Brake_FL");
            m_axles[0]->m_brake_right = chrono_types::make_shared<GD250_BrakeSimpleFront>("Brake_FR");
            m_axles[1]->m_brake_left = chrono_types::make_shared<GD250_BrakeSimpleFront>("Brake_RL");
            m_axles[1]->m_brake_right = chrono_types::make_shared<GD250_BrakeSimpleFront>("Brake_RR");
            break;
        case BrakeType::SHAFTS:
            m_axles[0]->m_brake_left = chrono_types::make_shared<GD250_BrakeShaftsFront>("Brake_FL");
            m_axles[0]->m_brake_right = chrono_types::make_shared<GD250_BrakeShaftsFront>("Brake_FR");
            m_axles[1]->m_brake_left = chrono_types::make_shared<GD250_BrakeShaftsFront>("Brake_RL");
            m_axles[1]->m_brake_right = chrono_types::make_shared<GD250_BrakeShaftsFront>("Brake_RR");
            break;
    }

    // Create the driveline
    if(m_low_range) {
        GetLog() << "Low Range Driveline selected: Vmax ca. 65 km/h\n";
        m_driveline = chrono_types::make_shared<GD250_Driveline4WD_LowRange>("Driveline");
    } else {
        GetLog() << "High Range Driveline selected: Vmax ca. 130 km/h\n";
        m_driveline = chrono_types::make_shared<GD250_Driveline4WD>("Driveline");
    }
}

GD250_Vehicle::~GD250_Vehicle() {}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void GD250_Vehicle::Initialize(const ChCoordsys<>& chassisPos, double chassisFwdVel) {
    // Initialize the chassis subsystem.
    m_chassis->Initialize(m_system, chassisPos, chassisFwdVel, WheeledCollisionFamily::CHASSIS);

    // Initialize the steering subsystem (specify the steering subsystem's frame relative to the chassis reference
    // frame).
    ChVector<> offset = ChVector<>(0, 0, 0);
    ChQuaternion<> rotation = Q_from_AngAxis(0, ChVector<>(0, 1, 0));
    m_steerings[0]->Initialize(m_chassis, offset, rotation);

    // Initialize the axle subsystems.
    m_axles[0]->Initialize(m_chassis, nullptr, m_steerings[0], ChVector<>(0, 0, 0), ChVector<>(0), 0.0, m_omega[0],
                           m_omega[1]);
    m_axles[1]->Initialize(m_chassis, nullptr, nullptr, ChVector<>(-2.4, 0, 0), ChVector<>(0), 0.0, m_omega[2],
                           m_omega[3]);

    // Initialize the driveline subsystem
    std::vector<int> driven_susp_indexes(m_driveline->GetNumDrivenAxles());
    driven_susp_indexes[0] = 1;
    driven_susp_indexes[1] = 1;
    m_driveline->Initialize(m_chassis, m_axles, driven_susp_indexes);

    // Invoke base class method
    ChWheeledVehicle::Initialize(chassisPos, chassisFwdVel);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
double GD250_Vehicle::GetSpringForce(int axle, VehicleSide side) const {
    if(m_kinematic) {
        if (axle == 0)
            return std::static_pointer_cast<ChToeBarGAxleSimple>(m_axles[axle]->m_suspension)->GetSpringForce(side);
        else
            return std::static_pointer_cast<ChGAxleSimple>(m_axles[axle]->m_suspension)->GetSpringForce(side);
    } else {
        if (axle == 0)
            return std::static_pointer_cast<ChToeBarGAxle>(m_axles[axle]->m_suspension)->GetSpringForce(side);
        else
            return std::static_pointer_cast<ChGAxle>(m_axles[axle]->m_suspension)->GetSpringForce(side);
    }
}

double GD250_Vehicle::GetSpringLength(int axle, VehicleSide side) const {
    if(m_kinematic) {
        if (axle == 0)
            return std::static_pointer_cast<ChToeBarGAxleSimple>(m_axles[axle]->m_suspension)->GetSpringLength(side);
        else
            return std::static_pointer_cast<ChGAxleSimple>(m_axles[axle]->m_suspension)->GetSpringLength(side);
    } else {
        if (axle == 0)
            return std::static_pointer_cast<ChToeBarGAxle>(m_axles[axle]->m_suspension)->GetSpringLength(side);
        else
            return std::static_pointer_cast<ChGAxle>(m_axles[axle]->m_suspension)->GetSpringLength(side);
    }
}

double GD250_Vehicle::GetSpringDeformation(int axle, VehicleSide side) const {
    if(m_kinematic) {
        if (axle == 0)
            return std::static_pointer_cast<ChToeBarGAxleSimple>(m_axles[axle]->m_suspension)->GetSpringDeformation(side);
        else
            return std::static_pointer_cast<ChGAxleSimple>(m_axles[axle]->m_suspension)->GetSpringDeformation(side);
    } else {
        if (axle == 0)
            return std::static_pointer_cast<ChToeBarGAxle>(m_axles[axle]->m_suspension)->GetSpringDeformation(side);
        else
            return std::static_pointer_cast<ChGAxle>(m_axles[axle]->m_suspension)->GetSpringDeformation(side);
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
double GD250_Vehicle::GetShockForce(int axle, VehicleSide side) const {
    if(m_kinematic) {
        if (axle == 0)
            return std::static_pointer_cast<ChToeBarGAxleSimple>(m_axles[axle]->m_suspension)->GetShockForce(side);
        else
            return std::static_pointer_cast<ChGAxleSimple>(m_axles[axle]->m_suspension)->GetShockForce(side);
    } else {
        if (axle == 0)
            return std::static_pointer_cast<ChToeBarGAxle>(m_axles[axle]->m_suspension)->GetShockForce(side);
        else
            return std::static_pointer_cast<ChGAxle>(m_axles[axle]->m_suspension)->GetShockForce(side);
    }
}

double GD250_Vehicle::GetShockLength(int axle, VehicleSide side) const {
    if(m_kinematic) {
        if (axle == 0)
            return std::static_pointer_cast<ChToeBarGAxleSimple>(m_axles[axle]->m_suspension)->GetShockLength(side);
        else
            return std::static_pointer_cast<ChGAxleSimple>(m_axles[axle]->m_suspension)->GetShockLength(side);
    } else {
        if (axle == 0)
            return std::static_pointer_cast<ChToeBarGAxle>(m_axles[axle]->m_suspension)->GetShockLength(side);
        else
            return std::static_pointer_cast<ChGAxle>(m_axles[axle]->m_suspension)->GetShockLength(side);
    }
}

double GD250_Vehicle::GetShockVelocity(int axle, VehicleSide side) const {
    if(m_kinematic) {
        if (axle == 0)
            return std::static_pointer_cast<ChToeBarGAxleSimple>(m_axles[axle]->m_suspension)->GetShockVelocity(side);
        else
            return std::static_pointer_cast<ChGAxleSimple>(m_axles[axle]->m_suspension)->GetShockVelocity(side);
    } else {
        if (axle == 0)
            return std::static_pointer_cast<ChToeBarGAxle>(m_axles[axle]->m_suspension)->GetShockVelocity(side);
        else
            return std::static_pointer_cast<ChGAxle>(m_axles[axle]->m_suspension)->GetShockVelocity(side);
    }
}

// -----------------------------------------------------------------------------
// Log the hardpoint locations for the front-right and rear-right suspension
// subsystems (display in inches)
// -----------------------------------------------------------------------------
void GD250_Vehicle::LogHardpointLocations() {
    GetLog().SetNumFormat("%7.3f");

    GetLog() << "\n---- FRONT suspension hardpoint locations (LEFT side)\n";
    std::static_pointer_cast<ChToeBarGAxle>(m_axles[0]->m_suspension)
        ->LogHardpointLocations(ChVector<>(0, 0, 0), false);

    GetLog() << "\n---- REAR suspension hardpoint locations (LEFT side)\n";
    std::static_pointer_cast<ChGAxle>(m_axles[1]->m_suspension)->LogHardpointLocations(ChVector<>(0, 0, 0), false);

    GetLog() << "\n\n";

    GetLog().SetNumFormat("%g");
}

// -----------------------------------------------------------------------------
// Log the spring length, deformation, and force.
// Log the shock length, velocity, and force.
// Log constraint violations of suspension joints.
//
// Lengths are reported in inches, velocities in inches/s, and forces in lbf
// -----------------------------------------------------------------------------
void GD250_Vehicle::DebugLog(int what) {
    GetLog().SetNumFormat("%10.2f");

    if (what & OUT_SPRINGS) {
        GetLog() << "\n---- Spring (front-left, front-right, rear-left, rear-right)\n";
        GetLog() << "Length [m]       " << GetSpringLength(0, LEFT) << "  " << GetSpringLength(0, RIGHT) << "  "
                 << GetSpringLength(1, LEFT) << "  " << GetSpringLength(1, RIGHT) << "\n";
        GetLog() << "Deformation [m]  " << GetSpringDeformation(0, LEFT) << "  " << GetSpringDeformation(0, RIGHT)
                 << "  " << GetSpringDeformation(1, LEFT) << "  " << GetSpringDeformation(1, RIGHT) << "\n";
        GetLog() << "Force [N]         " << GetSpringForce(0, LEFT) << "  " << GetSpringForce(0, RIGHT) << "  "
                 << GetSpringForce(1, LEFT) << "  " << GetSpringForce(1, RIGHT) << "\n";
    }

    if (what & OUT_SHOCKS) {
        GetLog() << "\n---- Shock (front-left, front-right, rear-left, rear-right)\n";
        GetLog() << "Length [m]       " << GetShockLength(0, LEFT) << "  " << GetShockLength(0, RIGHT) << "  "
                 << GetShockLength(1, LEFT) << "  " << GetShockLength(1, RIGHT) << "\n";
        GetLog() << "Velocity [m/s]   " << GetShockVelocity(0, LEFT) << "  " << GetShockVelocity(0, RIGHT) << "  "
                 << GetShockVelocity(1, LEFT) << "  " << GetShockVelocity(1, RIGHT) << "\n";
        GetLog() << "Force [N]         " << GetShockForce(0, LEFT) << "  " << GetShockForce(0, RIGHT) << "  "
                 << GetShockForce(1, LEFT) << "  " << GetShockForce(1, RIGHT) << "\n";
    }

    if (what & OUT_CONSTRAINTS) {
        // Report constraint violations for all joints
        LogConstraintViolations();
    }

    GetLog().SetNumFormat("%g");
}

}  // namespace gwagon
}  // end namespace vehicle
}  // end namespace chrono
