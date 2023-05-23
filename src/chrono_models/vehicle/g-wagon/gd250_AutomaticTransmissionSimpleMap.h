// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2023 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Rainer Gericke
// =============================================================================
//
// Automatic transmssion model for the G-wagon vehicle.
// - both power and torque limited
// - no torque converter
// - simple gear-shifting model (in automatic mode)
//
// =============================================================================

#ifndef GD250_AUTOMATIC_TRANSMISSION_SIMPLEMAP_H
#define GD250_AUTOMATIC_TRANSMISSION_SIMPLEMAP_H

#include "chrono_vehicle/powertrain/ChAutomaticTransmissionSimpleMap.h"

#include "chrono_models/ChApiModels.h"

namespace chrono {
namespace vehicle {
namespace gwagon {

/// @addtogroup vehicle_models_gwagon
/// @{

/// G-wagon automatic transmission model template based on a simple gear-shifting model.
class CH_MODELS_API GD250_AutomaticTransmissionSimpleMap : public ChAutomaticTransmissionSimpleMap {
  public:
    GD250_AutomaticTransmissionSimpleMap(const std::string& name);

    /// Set the transmission gear ratios (one or more forward gear ratios and a single reverse gear ratio).
    virtual void SetGearRatios(std::vector<double>& fwd, double& rev) override;

    /// Set the ideal shift points for automatic gear shifting.
    /// For each forward gear, specify a pair (min, max) with the minimum and
    /// maximum engine speed for shifting (down and up, respectively).
    virtual void SetShiftPoints(
        std::vector<std::pair<double, double>>& shift_bands  ///< [out] down-shift/up-shift points
        ) override;
};

/// @} vehicle_models_gwagon

}  // end namespace gwagon
}  // end namespace vehicle
}  // end namespace chrono

#endif
