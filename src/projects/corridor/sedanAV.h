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
// =============================================================================

#ifndef AV_SEDAN_H
#define AV_SEDAN_H

#include "chrono_models/vehicle/sedan/Sedan.h"

#include "vehicle.h"

namespace av {

class SedanAV : public Vehicle {
  public:
    ~SedanAV();

  private:
    SedanAV(Framework* framework, unsigned int id, const chrono::ChCoordsys<>& init_pos);

    virtual chrono::vehicle::ChWheeledVehicle& GetVehicle() const override;

    virtual double GetLookAheadDistance() const override { return 5; }
    virtual chrono::ChVector<> GetSteeringGainsPID() const override { return chrono::ChVector<>(0.8, 0, 0); }
    virtual chrono::ChVector<> GetSpeedGainsPID() const override { return chrono::ChVector<>(0.4, 0, 0); }

    virtual chrono::ChCoordsys<> GetLidarPosition() const override { return chrono::ChCoordsys<>({ 2, 0, 0 }, { 1, 0, 0, 0 }); }

    virtual void Synchronize(double time) override;
    virtual void Advance(double step) override;

    std::shared_ptr<chrono::vehicle::sedan::Sedan> m_sedan;

    friend class Framework;
};

}  // end namespace av

#endif
