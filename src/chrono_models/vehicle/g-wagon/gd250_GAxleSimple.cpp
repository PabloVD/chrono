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
// Authors: Radu Serban, Rainer Gericke
// =============================================================================
//
// Rear UAZBUS suspension subsystems (simple leafspring work a like).
//
// These concrete suspension subsystems are defined with respect to right-handed
// frames with X pointing towards the front, Y to the left, and Z up (as imposed
// by the base class ChDoubleWishbone) and origins at the midpoint between the
// lower control arms' connection points to the chassis.
//
// All point locations are provided for the left half of the suspension.
//
// =============================================================================

#include "chrono_models/vehicle/g-wagon/gd250_GAxleSimple.h"

namespace chrono {
namespace vehicle {
namespace gwagon {

// -----------------------------------------------------------------------------
// Static variables
// -----------------------------------------------------------------------------

const double GD250_GAxleSimple::m_axleTubeMass = 124.0;
const double GD250_GAxleSimple::m_spindleMass = 14.705;
const double GD250_GAxleSimple::m_panhardRodMass = 10.0;

const double GD250_GAxleSimple::m_axleTubeRadius = 0.0476;
const double GD250_GAxleSimple::m_spindleRadius = 0.10;
const double GD250_GAxleSimple::m_spindleWidth = 0.06;
const double GD250_GAxleSimple::m_panhardRodRadius = 0.03;

const ChVector<> GD250_GAxleSimple::m_axleTubeInertia(22.21, 0.0775, 22.21);
const ChVector<> GD250_GAxleSimple::m_spindleInertia(0.04117, 0.07352, 0.04117);
const ChVector<> GD250_GAxleSimple::m_panhardRodInertia(1.0, 0.04, 1.0);

const double GD250_GAxleSimple::m_springDesignLength = 0.3;
const double GD250_GAxleSimple::m_springCoefficient = 102328.0584;
const double GD250_GAxleSimple::m_springRestLength = m_springDesignLength + 0.03;
const double GD250_GAxleSimple::m_springMinLength = m_springDesignLength - 0.08;
const double GD250_GAxleSimple::m_springMaxLength = m_springDesignLength + 0.08;
const double GD250_GAxleSimple::m_damperCoefficient = 18095.57368;
const double GD250_GAxleSimple::m_damperDegressivityCompression = 3.0;
const double GD250_GAxleSimple::m_damperDegressivityExpansion = 1.0;
const double GD250_GAxleSimple::m_axleShaftInertia = 0.4;

// ---------------------------------------------------------------------------------------
// UAZBUS spring functor class - implements a linear spring + bump stop + rebound stop
// ---------------------------------------------------------------------------------------
class GD250_SpringForceRear : public ChLinkTSDA::ForceFunctor {
  public:
    GD250_SpringForceRear(double spring_constant, double min_length, double max_length);

    virtual double evaluate(double time, double rest_length, double length, double vel, ChLinkTSDA* link) override;

  private:
    double m_spring_constant;
    double m_min_length;
    double m_max_length;

    ChFunction_Recorder m_bump;
};

GD250_SpringForceRear::GD250_SpringForceRear(double spring_constant, double min_length, double max_length)
    : m_spring_constant(spring_constant), m_min_length(min_length), m_max_length(max_length) {
    // From ADAMS/Car
    m_bump.AddPoint(0.0, 0.0);
    m_bump.AddPoint(2.0e-3, 200.0);
    m_bump.AddPoint(4.0e-3, 400.0);
    m_bump.AddPoint(6.0e-3, 600.0);
    m_bump.AddPoint(8.0e-3, 800.0);
    m_bump.AddPoint(10.0e-3, 1000.0);
    m_bump.AddPoint(20.0e-3, 2500.0);
    m_bump.AddPoint(30.0e-3, 4500.0);
    m_bump.AddPoint(40.0e-3, 7500.0);
    m_bump.AddPoint(50.0e-3, 12500.0);
}

double GD250_SpringForceRear::evaluate(double time, double rest_length, double length, double vel, ChLinkTSDA* link) {
    double force = 0;

    double defl_spring = rest_length - length;
    double defl_bump = 0.0;
    double defl_rebound = 0.0;

    if (length < m_min_length) {
        defl_bump = m_min_length - length;
    }

    if (length > m_max_length) {
        defl_rebound = length - m_max_length;
    }

    force = defl_spring * m_spring_constant + m_bump.Get_y(defl_bump) - m_bump.Get_y(defl_rebound);

    return force;
}

// -----------------------------------------------------------------------------
// UAZBUS shock functor class - implements a nonlinear damper
// -----------------------------------------------------------------------------
class UAZBUS_ShockForceRear : public ChLinkTSDA::ForceFunctor {
  public:
    UAZBUS_ShockForceRear(double compression_slope,
                          double compression_degressivity,
                          double expansion_slope,
                          double expansion_degressivity);

    virtual double evaluate(double time, double rest_length, double length, double vel, ChLinkTSDA* link) override;

  private:
    double m_slope_compr;
    double m_slope_expand;
    double m_degres_compr;
    double m_degres_expand;
};

UAZBUS_ShockForceRear::UAZBUS_ShockForceRear(double compression_slope,
                                             double compression_degressivity,
                                             double expansion_slope,
                                             double expansion_degressivity)
    : m_slope_compr(compression_slope),
      m_degres_compr(compression_degressivity),
      m_slope_expand(expansion_slope),
      m_degres_expand(expansion_degressivity) {}

double UAZBUS_ShockForceRear::evaluate(double time, double rest_length, double length, double vel, ChLinkTSDA* link) {
    // Simple model of a degressive damping characteristic
    double force = 0;

    // Calculate Damping Force
    if (vel >= 0) {
        force = -m_slope_expand / (1.0 + m_degres_expand * std::abs(vel)) * vel;
    } else {
        force = -m_slope_compr / (1.0 + m_degres_compr * std::abs(vel)) * vel;
    }

    return force;
}

GD250_GAxleSimple::GD250_GAxleSimple(const std::string& name) : ChGAxleSimple(name) {
    m_springForceCB =
        chrono_types::make_shared<GD250_SpringForceRear>(m_springCoefficient, m_springMinLength, m_springMaxLength);

    m_shockForceCB = chrono_types::make_shared<UAZBUS_ShockForceRear>(
        m_damperCoefficient, m_damperDegressivityCompression, m_damperCoefficient, m_damperDegressivityExpansion);
}

// -----------------------------------------------------------------------------
// Destructors
// -----------------------------------------------------------------------------
GD250_GAxleSimple::~GD250_GAxleSimple() {}

const ChVector<> GD250_GAxleSimple::getLocation(PointId which) {
    switch (which) {
        case SPRING_A:
            return ChVector<>(0.0, 0.5142, m_axleTubeRadius);
        case SPRING_C:
            return ChVector<>(0.0, 0.5142, m_axleTubeRadius + m_springDesignLength);
        case SHOCK_A:
            return ChVector<>(-0.125, 0.441, -0.0507);
        case SHOCK_C:
            return ChVector<>(-0.3648, 0.4193, 0.4298);
        case SPINDLE:
            return ChVector<>(0.0, 0.7325, 0.0);
        case PANHARD_A:
            return ChVector<>(-0.15, -0.5142, 0.0);
        case PANHARD_C:
            return ChVector<>(-0.15, 0.5142, 0.0);
        default:
            return ChVector<>(0, 0, 0);
    }
}

}  // namespace gwagon
}  // end namespace vehicle
}  // end namespace chrono
