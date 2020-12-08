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
// M113 vehicle model.
//
// =============================================================================

#ifndef M113a_VEHICLE_H
#define M113a_VEHICLE_H

#include "chrono_vehicle/tracked_vehicle/ChTrackedVehicle.h"

#include "chrono_models/ChApiModels.h"
#include "chrono_models/vehicle/ChVehicleModelDefs.h"

namespace chrono {
namespace vehicle {
namespace m113 {

class CH_MODELS_API M113a_Vehicle : public ChTrackedVehicle {
  public:
    M113a_Vehicle(bool fixed,
                  ChContactMethod contact_method = ChContactMethod::NSC,
                  CollisionType chassis_collision_type = CollisionType::NONE);

    M113a_Vehicle(bool fixed, ChSystem* system, CollisionType chassis_collision_type = CollisionType::NONE);

    ~M113a_Vehicle() {}

    virtual void Initialize(const ChCoordsys<>& chassisPos, double chassisFwdVel = 0) override;

  private:
    void Create(bool fixed, CollisionType chassis_collision_type);
};

}  // end namespace m113
}  // end namespace vehicle
}  // end namespace chrono

#endif
