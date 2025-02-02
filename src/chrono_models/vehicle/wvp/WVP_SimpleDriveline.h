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
// Authors: Radu Serban, Asher ELmquist
// =============================================================================
//
// Simple driveline model. This is a hardcoded driveline that represents
// the WVP 4WD driveline. It is for temporary use as the ChaftsDriveline
// seems to generate excess noise and the simpledriveline template does not
// connect properly to the simeplePowertrain model.
//
// =============================================================================

#ifndef WVP_SIMPLE_DRIVELINE_H
#define WVP_SIMPLE_DRIVELINE_H

#include "chrono_vehicle/ChApiVehicle.h"
#include "chrono_vehicle/wheeled_vehicle/ChDrivelineWV.h"

#include "chrono_models/ChApiModels.h"

namespace chrono {
namespace vehicle {
namespace wvp {

/// @addtogroup vehicle_models_wvp
/// @{

/// Simple driveline model. This template can be used to model a 4WD driveline.
/// It uses a constant front/rear torque split (a value between 0 and 1) and a
/// simple model for Torsen limited-slip differentials.
class CH_MODELS_API WVP_SimpleDriveline : public ChDrivelineWV {
  public:
    WVP_SimpleDriveline(const std::string& name);

    virtual ~WVP_SimpleDriveline() {}

    /// Get the name of the vehicle subsystem template.
    virtual std::string GetTemplateName() const override { return "WVP_SimpleDriveline"; }

    /// Return the number of driven axles.
    virtual int GetNumDrivenAxles() const final override { return 2; }

    /// Initialize the driveline subsystem.
    /// This function connects this driveline subsystem to the specified axle subsystems.
    virtual void Initialize(std::shared_ptr<ChChassis> chassis,   ///< associated chassis subsystem
                            const ChAxleList& axles,              ///< list of all vehicle axle subsystems
                            const std::vector<int>& driven_axles  ///< indexes of the driven vehicle axles
                            ) override;

    /// Update the driveline subsystem: apply the specified motor torque.
    /// This represents the input to the driveline subsystem from the powertrain system.
    virtual void Synchronize(double time, const DriverInputs& driver_inputs, double torque) override;

    /// Get the motor torque to be applied to the specified spindle.
    virtual double GetSpindleTorque(int axle, VehicleSide side) const override;

    /// Return the output driveline speed of the driveshaft.
    /// This represents the output from the driveline subsystem that is passed to the transmission subsystem.
    virtual double GetOutputDriveshaftSpeed() const override { return m_driveshaft_speed; }

  private:
    /// he front torque fraction [0,1].
    double m_frontTorqueFraction = 0.5;
    /// the torque bias ratio for the front differential.
    double m_frontDifferentialMaxBias = 2.5;
    /// the torque bias ratio for the rear differential.
    double m_rearDifferentialMaxBias = 2.5;

    double m_transferRatio = 1;
    double m_gearHubReduction = 2.0;
    double m_diffGearReduction = 2.69;

    bool m_diffLockCenter = false;
    bool m_diffLockFront = false;
    bool m_diffLockRear = false;

    double m_driveshaft_speed;  ///< output to transmisson

    std::shared_ptr<ChShaft> m_front_left;
    std::shared_ptr<ChShaft> m_front_right;
    std::shared_ptr<ChShaft> m_rear_left;
    std::shared_ptr<ChShaft> m_rear_right;
};

/// @} vehicle_wheeled_driveline

}  // end namespace vehicle
}  // end namespace chrono
}  // end namespace wvp
#endif
