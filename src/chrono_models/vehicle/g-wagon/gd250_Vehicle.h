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
// Base class for the GD250 vehicle models
//
// =============================================================================

#ifndef GD250_VEHICLE_H
#define GD250_VEHICLE_H

#include "chrono_vehicle/wheeled_vehicle/ChWheeledVehicle.h"

#include "chrono_models/ChApiModels.h"
#include "chrono_models/vehicle/ChVehicleModelDefs.h"

namespace chrono {
    namespace vehicle {
        namespace gwagon {

/// @addtogroup vehicle_models_uaz
/// @{

/// UAZ vehicle system.
            class CH_MODELS_API GD250_Vehicle : public ChWheeledVehicle {
            public:
                GD250_Vehicle(const bool fixed,
                        BrakeType brake_type,
                        SteeringTypeWV steering_model,
                        ChContactMethod contact_method = ChContactMethod::NSC,
                        CollisionType chassis_collision_type = CollisionType::NONE,
                        bool kinematic_mode=true,
                        bool low_range = false);

                GD250_Vehicle(ChSystem* system,
                        const bool fixed,
                        BrakeType brake_type,
                        SteeringTypeWV steering_model,
                        CollisionType chassis_collision_type = CollisionType::NONE,
                        bool kinematic_mode=true,
                        bool low_range = false);

                ~GD250_Vehicle();

                virtual int GetNumberAxles() const override { return 2; }

                virtual double GetWheelbase() const override { return 2.4; }
                virtual double GetMinTurningRadius() const override { return 5.8; }
                virtual double GetMaxSteeringAngle() const override { return 27 * CH_C_DEG_TO_RAD; }

                void SetInitWheelAngVel(const std::vector<double>& omega) {
                    assert(omega.size() == 4);
                    m_omega = omega;
                }

                double GetSpringForce(int axle, VehicleSide side) const;
                double GetSpringLength(int axle, VehicleSide side) const;
                double GetSpringDeformation(int axle, VehicleSide side) const;

                double GetShockForce(int axle, VehicleSide side) const;
                double GetShockLength(int axle, VehicleSide side) const;
                double GetShockVelocity(int axle, VehicleSide side) const;

                virtual void Initialize(const ChCoordsys<>& chassisPos, double chassisFwdVel = 0) override;

                // Log debugging information
                void LogHardpointLocations();  /// suspension hardpoints at design
                void DebugLog(int what);       /// shock forces and lengths, constraints, etc.

            private:
                void Create(bool fixed, BrakeType brake_type, SteeringTypeWV steering_model, CollisionType chassis_collision_type);

                std::vector<double> m_omega;
                bool m_kinematic;
                bool m_low_range;
            };

/// @} vehicle_models_gd250

        }  // end namespace gwagon
    }  // end namespace vehicle
}  // end namespace chrono

#endif
