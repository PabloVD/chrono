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

#include "traffic_light.h"
#include "framework.h"

using namespace chrono;

namespace av {

TrafficLightList TrafficLight::m_traffic_lights;

// -----------------------------------------------------------------------------

TrafficLight::TrafficLight(Framework* framework,
                           unsigned int id,
                           const chrono::ChVector<>& center,
                           double radius,
                           const chrono::ChCoordsys<>& pos)
    : Agent(framework, id), m_center(center), m_radius(radius), m_pos(pos) {
    m_body = std::shared_ptr<ChBody>(framework->m_system->NewBody());
    m_body->SetPos(pos.pos);
    m_body->SetRot(pos.rot);
    m_body->SetBodyFixed(true);
    m_body->SetCollide(false);

    auto cyl = std::make_shared<ChCylinderShape>();
    cyl->GetCylinderGeometry().p1 = ChVector<>(0, 0, 0);
    cyl->GetCylinderGeometry().p2 = ChVector<>(0, 0, 3);
    cyl->GetCylinderGeometry().rad = 0.2;
    m_body->AddAsset(cyl);

    auto sphere = std::make_shared<ChSphereShape>();
    sphere->GetSphereGeometry().rad = 0.4;
    sphere->Pos = ChVector<>(0, 0, 3);
    m_body->AddAsset(sphere);

    m_body->AddAsset(std::make_shared<ChColorAsset>(ChColor(0.6f, 0, 0)));

    framework->m_system->AddBody(m_body);

    // Prepare messages sent by this agent
    m_map_msg = std::make_shared<MessageMAP>();
    m_map_msg->type = Message::MAP;
    m_map_msg->senderID = id;
    m_map_msg->time = 0;
    m_map_msg->intersectionID = 9999;
    m_map_msg->num_lanes = 3;

    m_spat_msg = std::make_shared<MessageSPAT>();
    m_spat_msg->type = Message::SPAT;
    m_spat_msg->senderID = id;
    m_spat_msg->time = 0;
    m_spat_msg->phase = 1;
    m_spat_msg->time_phase = 0;

    //// TODO: provide mechanism for setting this frequency
    m_bcast_freq = 2;
}

TrafficLight::~TrafficLight() {}

std::shared_ptr<TrafficLight> TrafficLight::Find(unsigned int id) {
    auto it = m_traffic_lights.find(id);
    if (it != m_traffic_lights.end())
        return it->second;
    return nullptr;
}

void TrafficLight::Broadcast(double time) {
    m_map_msg->time = static_cast<float>(time);

    m_spat_msg->time = static_cast<float>(time);

    for (auto v : Vehicle::GetList()) {
        if ((GetPosition().pos - v.second->GetPosition().pos).Length() <= 1000) {
            Send(v.second, m_map_msg);
            Send(v.second, m_spat_msg);
        }
    }
}

void TrafficLight::Unicast(double time) {
    //// TODO
}

void TrafficLight::ProcessMessages() {
    while (!m_messages.empty()) {
        auto msg = m_messages.front();
        if (m_framework->Verbose()) {
            std::cout << "Traffic Light " << m_id << " received message from " << msg->senderID
                      << " msg type: " << msg->type << " time stamp: " << msg->time << std::endl;
        }
        switch (msg->type) {
            case Message::VEH:
                ProcessMessageVEH(std::static_pointer_cast<MessageVEH>(msg));
                break;
            default:
                break;
        }
        m_messages.pop();
    }
}

void TrafficLight::ProcessMessageVEH(std::shared_ptr<MessageVEH> msg) {
    //// TODO
    if (m_framework->Verbose()) {
        auto& loc = msg->location;
        std::cout << "      Vehicle location: " << loc.x() << " " << loc.y() << " " << loc.z() << std::endl;
    }
}

void TrafficLight::Synchronize(double time) {
    //// TODO:
    //// - implement FSM for this agent
    //// - update SPAT message for next broadcast
}

void TrafficLight::Advance(double step) {
    //// TODO
}

}  // end namespace av
