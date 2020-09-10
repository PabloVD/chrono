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
// Authors: Rainer Gericke
// =============================================================================
// Header for an helper class defining materials for Phong shading
// =============================================================================

#ifndef CH_VSG_PHONG_MATERIAL_H
#define CH_VSG_PHONG_MATERIAL_H

#include <iostream>
#include "chrono/core/ChVector.h"
#include "chrono/physics/ChSystemNSC.h"
#include "chrono_vsg/core/ChApiVSG.h"
#include "chrono_vsg/resources/ChVSGSettings.h"

#include <vsg/all.h>

namespace chrono {
namespace vsg3d {

class CH_VSG_API ChVSGPhongMaterial {
  public:
    ChVSGPhongMaterial(PhongPresets mat = PhongPresets::Emerald);

    vsg::vec3 ambientColor;
    vsg::vec3 diffuseColor;
    vsg::vec3 specularColor;
    float shininess;
    float opacity;
};
}  // namespace vsg3d
}  // namespace chrono
#endif
