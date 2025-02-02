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
// Authors: Radu Serban, Justin Madsen, Daniel Melanz
// =============================================================================
//
// Front and Rear UAZBUS suspension subsystems (double A-arm).
//
// These concrete suspension subsystems are defined with respect to right-handed
// frames with X pointing towards the front, Y to the left, and Z up (as imposed
// by the base class ChDoubleWishbone) and origins at the midpoint between the
// lower control arms' connection points to the chassis.
//
// All point locations are provided for the left half of the suspension.
//
// =============================================================================

#include "chrono_models/vehicle/g-wagon/gd250_ToeBarGAxleSimple.h"

namespace chrono {
namespace vehicle {
namespace gwagon {

// -----------------------------------------------------------------------------
// Static variables
// -----------------------------------------------------------------------------

const double GD250_ToeBarGAxleSimple::m_axleTubeMass = 124.0;
const double GD250_ToeBarGAxleSimple::m_panhardRodMass = 10.0;
const double GD250_ToeBarGAxleSimple::m_arbMass = 5.0;
const double GD250_ToeBarGAxleSimple::m_spindleMass = 14.705;
const double GD250_ToeBarGAxleSimple::m_knuckleMass = 10.0;
const double GD250_ToeBarGAxleSimple::m_tierodMass = 5.0;
const double GD250_ToeBarGAxleSimple::m_draglinkMass = 5.0;

const double GD250_ToeBarGAxleSimple::m_axleTubeRadius = 0.0476;
const double GD250_ToeBarGAxleSimple::m_panhardRodRadius = 0.03;
const double GD250_ToeBarGAxleSimple::m_arbRadius = 0.025;
const double GD250_ToeBarGAxleSimple::m_spindleRadius = 0.10;
const double GD250_ToeBarGAxleSimple::m_spindleWidth = 0.06;
const double GD250_ToeBarGAxleSimple::m_knuckleRadius = 0.05;
const double GD250_ToeBarGAxleSimple::m_tierodRadius = 0.02;
const double GD250_ToeBarGAxleSimple::m_draglinkRadius = 0.02;

const ChVector<> GD250_ToeBarGAxleSimple::m_axleTubeInertia(22.21, 0.0775, 22.21);
const ChVector<> GD250_ToeBarGAxleSimple::m_panhardRodInertia(1.0, 0.04, 1.0);
const ChVector<> GD250_ToeBarGAxleSimple::m_arbInertia(0.5, 0.02, 0.5);
const ChVector<> GD250_ToeBarGAxleSimple::m_spindleInertia(0.04117, 0.07352, 0.04117);
const ChVector<> GD250_ToeBarGAxleSimple::m_knuckleInertia(0.1, 0.1, 0.1);
const ChVector<> GD250_ToeBarGAxleSimple::m_tierodInertia(1.0, 0.1, 1.0);
const ChVector<> GD250_ToeBarGAxleSimple::m_draglinkInertia(0.1, 1.0, 0.1);

const double GD250_ToeBarGAxleSimple::m_arb_stiffness = 1000.0;
const double GD250_ToeBarGAxleSimple::m_arb_damping = 10.0;

const double GD250_ToeBarGAxleSimple::m_springDesignLength = 0.3;
const double GD250_ToeBarGAxleSimple::m_springCoefficient = 76746.04382;
const double GD250_ToeBarGAxleSimple::m_springRestLength = m_springDesignLength + 0.0621225507207084;
const double GD250_ToeBarGAxleSimple::m_springMinLength = m_springDesignLength - 0.08;
const double GD250_ToeBarGAxleSimple::m_springMaxLength = m_springDesignLength + 0.08;
const double GD250_ToeBarGAxleSimple::m_damperCoefficient = 19193.25429;
const double GD250_ToeBarGAxleSimple::m_damperDegressivityCompression = 3.0;
const double GD250_ToeBarGAxleSimple::m_damperDegressivityExpansion = 1.0;
const double GD250_ToeBarGAxleSimple::m_axleShaftInertia = 0.4;

// ---------------------------------------------------------------------------------------
// GD250 spring functor class - implements a linear spring + bump stop + rebound stop
// ---------------------------------------------------------------------------------------
class GD250_SpringForceFront : public ChLinkTSDA::ForceFunctor {
  public:
    GD250_SpringForceFront(double spring_constant, double min_length, double max_length);

    virtual double evaluate(double time,
                            double rest_length,
                            double length,
                            double vel,
                            const ChLinkTSDA& link) override;

  private:
    double m_spring_constant;
    double m_min_length;
    double m_max_length;

    ChFunction_Recorder m_bump;
};

GD250_SpringForceFront::GD250_SpringForceFront(double spring_constant, double min_length, double max_length)
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

double GD250_SpringForceFront::evaluate(double time,
                                        double rest_length,
                                        double length,
                                        double vel,
                                        const ChLinkTSDA& link) {
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
// GD250 shock functor class - implements a nonlinear damper
// -----------------------------------------------------------------------------
class GD250_ShockForceFront : public ChLinkTSDA::ForceFunctor {
  public:
    GD250_ShockForceFront(double compression_slope,
                          double compression_degressivity,
                          double expansion_slope,
                          double expansion_degressivity);

    virtual double evaluate(double time,
                            double rest_length,
                            double length,
                            double vel,
                            const ChLinkTSDA& link) override;

  private:
    double m_slope_compr;
    double m_slope_expand;
    double m_degres_compr;
    double m_degres_expand;
};

GD250_ShockForceFront::GD250_ShockForceFront(double compression_slope,
                                             double compression_degressivity,
                                             double expansion_slope,
                                             double expansion_degressivity)
    : m_slope_compr(compression_slope),
      m_degres_compr(compression_degressivity),
      m_slope_expand(expansion_slope),
      m_degres_expand(expansion_degressivity) {}

double GD250_ShockForceFront::evaluate(double time,
                                       double rest_length,
                                       double length,
                                       double vel,
                                       const ChLinkTSDA& link) {
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

GD250_ToeBarGAxleSimple::GD250_ToeBarGAxleSimple(const std::string& name) : ChToeBarGAxleSimple(name) {
    m_springForceCB =
        chrono_types::make_shared<GD250_SpringForceFront>(m_springCoefficient, m_springMinLength, m_springMaxLength);

    m_shockForceCB = chrono_types::make_shared<GD250_ShockForceFront>(
        m_damperCoefficient, m_damperDegressivityCompression, m_damperCoefficient, m_damperDegressivityExpansion);
}

// -----------------------------------------------------------------------------
// Destructors
// -----------------------------------------------------------------------------
GD250_ToeBarGAxleSimple::~GD250_ToeBarGAxleSimple() {}

const ChVector<> GD250_ToeBarGAxleSimple::getLocation(PointId which) {
    switch (which) {
        case SPRING_A:
            return ChVector<>(0.0, 0.3824, m_axleTubeRadius);
        case SPRING_C:
            return ChVector<>(0.0, 0.3824, m_axleTubeRadius + m_springDesignLength);
        case SHOCK_A:
            return ChVector<>(-0.1, 0.441, -0.0507);
        case SHOCK_C:
            return ChVector<>(-0.2, 0.4193, 0.4298);
        case SPINDLE:
            return ChVector<>(0.0, 0.7325, 0.0);
        case KNUCKLE_CM:
            return ChVector<>(0.0, 0.7325 - 0.07, 0.0);
        case KNUCKLE_L:
            return ChVector<>(0.0, 0.7325 - 0.07 + 0.0098058067569092, -0.1);
        case KNUCKLE_U:
            return ChVector<>(0.0, 0.7325 - 0.07 - 0.0098058067569092, 0.1);
        case KNUCKLE_DRL:
            return ChVector<>(0.2, 0.7, 0.1);
        case TIEROD_K:
            return ChVector<>(-0.190568826619798, 0.7325 - 0.07 - 0.060692028477827, 0.1);
        case DRAGLINK_C:
            return ChVector<>(0.2, 0.2, 0.1);
        case PANHARD_A:
            return ChVector<>(0.1, -0.44, 0.0);
        case PANHARD_C:
            return ChVector<>(0.1, 0.44, 0.0);
        case ANTIROLL_A:
            return ChVector<>(0.0, 0.35, -0.05);
        case ANTIROLL_C:
            return ChVector<>(-0.4, 0.35, -0.05);
        default:
            return ChVector<>(0, 0, 0);
    }
}

}  // namespace gwagon
}  // end namespace vehicle
}  // end namespace chrono
