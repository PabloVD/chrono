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
// Base class for the WVP vehicle models
//
// =============================================================================

#ifndef WVP_VEHICLE_H
#define WVP_VEHICLE_H

#include <vector>

#include "chrono/core/ChCoordsys.h"
#include "chrono/physics/ChMaterialSurface.h"
#include "chrono/physics/ChSystem.h"

#include "chrono_vehicle/wheeled_vehicle/ChWheeledVehicle.h"

#include "chrono_models/ChApiModels.h"
#include "chrono_models/vehicle/ChVehicleModelDefs.h"

#include "chrono_models/vehicle/wvp/WVP_Chassis.h"
#include "chrono_models/vehicle/wvp/WVP_BrakeSimple.h"
#include "chrono_models/vehicle/wvp/WVP_DoubleWishbone.h"
#include "chrono_models/vehicle/wvp/WVP_Driveline4WD.h"
#include "chrono_models/vehicle/wvp/WVP_SimpleDriveline.h"
#include "chrono_models/vehicle/wvp/WVP_PitmanArm.h"
#include "chrono_models/vehicle/wvp/WVP_PitmanArmShafts.h"
#include "chrono_models/vehicle/wvp/WVP_Wheel.h"

namespace chrono {
namespace vehicle {
namespace wvp {

class CH_MODELS_API WVP_Vehicle : public ChWheeledVehicle {
  public:
    WVP_Vehicle(const bool fixed = false,
                SteeringTypeWV steering_model = SteeringTypeWV::PITMAN_ARM,
                ChContactMethod contact_method = ChContactMethod::NSC,
                CollisionType chassis_collision_type = CollisionType::NONE);

    WVP_Vehicle(ChSystem* system,
                const bool fixed = false,
                SteeringTypeWV steering_model = SteeringTypeWV::PITMAN_ARM,
                CollisionType chassis_collision_type = CollisionType::NONE);

    ~WVP_Vehicle();

    virtual int GetNumberAxles() const override { return 2; }

    virtual double GetWheelbase() const override { return 4.039; }
    virtual double GetMinTurningRadius() const override { return 9.11; }
    virtual double GetMaxSteeringAngle() const override { return 30.0 * CH_C_DEG_TO_RAD; }

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
    void Create(bool fixed, SteeringTypeWV steering_model, CollisionType chassis_collision_type);

    std::vector<double> m_omega;
};

}  // end namespace wvp
}  // end namespace vehicle
}  // end namespace chrono

#endif
