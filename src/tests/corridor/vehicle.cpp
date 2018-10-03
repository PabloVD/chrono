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
// =============================================================================

#include "vehicle.h"
#include "framework.h"
#include "traffic_light.h"

using namespace chrono;
using namespace chrono::vehicle;

namespace av {

const double Vehicle::m_throttle_threshold = 0.2;
const std::string Vehicle::m_types[] = {"TRUCK", "SEDAN", "VAN", "BUS"};
VehicleList Vehicle::m_vehicles;

// -----------------------------------------------------------------------------

Vehicle::Vehicle(Framework* framework, unsigned int id, Type vehicle_type)
    : Agent(framework, id),
      m_vehicle_type(vehicle_type),
      m_steeringPID(nullptr),
      m_speedPID(nullptr),
      m_steering(0),
      m_throttle(0),
      m_braking(0),
      m_MAP_id(0),
      m_SPAT_phase(-1),
      m_SPAT_time(0) {
    // Prepare messages sent by this agent
    m_vehicle_msg = std::make_shared<MessageVEH>();
    m_vehicle_msg->type = Message::VEH;
    m_vehicle_msg->senderID = id;
    m_vehicle_msg->time = 0;
    m_vehicle_msg->location = ChVector<>(0, 0, 0);
}

Vehicle::~Vehicle() {
    delete m_steeringPID;
    delete m_speedPID;
}

ChCoordsys<> Vehicle::GetPosition() const {
    return ChCoordsys<>(GetVehicle().GetVehiclePos(), GetVehicle().GetVehicleRot());
}

std::shared_ptr<Vehicle> Vehicle::Find(unsigned int id) {
    auto it = m_vehicles.find(id);
    if (it != m_vehicles.end())
        return it->second;
    return nullptr;
}

void Vehicle::SetupDriver(std::shared_ptr<chrono::ChBezierCurve> curve, bool closed, double target_speed) {
    m_steeringPID = new ChPathSteeringController(curve, closed);
    m_speedPID = new ChSpeedController();
    m_target_speed = target_speed;

    double dist = GetLookAheadDistance();
    ChVector<> steering_gains = GetSteeringGainsPID();
    ChVector<> speed_gains = GetSpeedGainsPID();
    m_steeringPID->SetLookAheadDistance(dist);
    m_steeringPID->SetGains(steering_gains.x(), steering_gains.y(), steering_gains.z());
    m_speedPID->SetGains(speed_gains.x(), speed_gains.y(), speed_gains.z());

    m_steeringPID->Reset(GetVehicle());
    m_speedPID->Reset(GetVehicle());
}

void Vehicle::AdvanceDriver(double step) {
    // Set the throttle and braking values based on the output from the speed controller.
    double out_speed = m_speedPID->Advance(GetVehicle(), m_target_speed, step);
    ChClampValue(out_speed, -1.0, 1.0);

    if (out_speed > 0) {
        // Vehicle moving too slow
        m_braking = 0;
        m_throttle = out_speed;
    } else if (m_throttle > m_throttle_threshold) {
        // Vehicle moving too fast: reduce throttle
        m_braking = 0;
        m_throttle = 1 + out_speed;
    } else {
        // Vehicle moving too fast: apply brakes
        m_braking = -out_speed;
        m_throttle = 0;
    }

    // Set the steering value based on the output from the steering controller.
    double out_steering = m_steeringPID->Advance(GetVehicle(), step);
    ChClampValue(out_steering, -1.0, 1.0);
    m_steering = out_steering;
}

void Vehicle::Broadcast(double time) {
    m_vehicle_msg->time = static_cast<float>(time);
    m_vehicle_msg->location = GetPosition().pos;

    for (auto a : Agent::GetList()) {
        if (a.second->GetId() != m_id && (GetPosition().pos - a.second->GetPosition().pos).Length() <= m_bcast_radius) {
            Send(a.second, m_vehicle_msg);
        }
    }
}

void Vehicle::Unicast(double time) {
    //// TODO
}

void Vehicle::ProcessMessages() {
    while (!m_messages.empty()) {
        auto msg = m_messages.front();
        if (m_framework->Verbose()) {
            std::cout << GetTypeName() << " " << m_id << " received message from " << msg->senderID
                      << " msg type: " << msg->type << " time stamp: " << msg->time << std::endl;
        }
        switch (msg->type) {
            case Message::MAP:
                ProcessMessageMAP(std::static_pointer_cast<MessageMAP>(msg));
                break;
            case Message::SPAT:
                ProcessMessageSPAT(std::static_pointer_cast<MessageSPAT>(msg));
                break;
            case Message::VEH:
                ProcessMessageVEH(std::static_pointer_cast<MessageVEH>(msg));
                break;
            default:
                break;
        }
        m_messages.pop();
    }
}

void Vehicle::ProcessMessageMAP(std::shared_ptr<MessageMAP> msg) {
    //// TODO
    m_MAP_id = msg->intersectionID;

    if (m_framework->Verbose()) {
        std::cout << "      Intersection ID: " << msg->intersectionID << std::endl;
    }
}

void Vehicle::ProcessMessageSPAT(std::shared_ptr<MessageSPAT> msg) {
    //// TODO
    m_SPAT_phase = msg->phase;
    m_SPAT_time = msg->time_phase;

    if (m_framework->Verbose()) {
        std::cout << "      Signal phase: " << msg->phase << "  Signal timing: " << msg->time_phase << std::endl;
    }
}

void Vehicle::ProcessMessageVEH(std::shared_ptr<MessageVEH> msg) {
    //// TODO
}

}  // end namespace av
