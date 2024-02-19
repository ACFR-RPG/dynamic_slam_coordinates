/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:
 
 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.
 
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */



#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>

namespace dsc
{

/**
 * @brief A landmark 3D projection factor
 * The factor contains the difference between the expecetd landmark measurement  and the measured ^k l^i_k
 * The error vector will be ^0 X_k^-1 * ^0 l^i_k - ^k l^i_k
 */
class Point3DFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>
{
private:
  gtsam::Point3 measured_;  // in camera coordinates

public:
  typedef boost::shared_ptr<Point3DFactor> shared_ptr;
  int is_dynamic_{ 0 };
  /**
   * @brief Construct a new Point 3 D Factor object
   *
   * @param poseKey
   * @param pointKey
   * @param m Measurement in the camera frame
   * @param model
   */
  Point3DFactor(gtsam::Key poseKey, gtsam::Key pointKey, const gtsam::Point3& m, gtsam::SharedNoiseModel model);

  gtsam::NonlinearFactor::shared_ptr clone() const override
  {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new Point3DFactor(*this)));
  }

  // error function
  // L is landmark in world coordinates
  gtsam::Vector evaluateError(const gtsam::Pose3& X, const gtsam::Point3& l,
                              boost::optional<gtsam::Matrix&> J1 = boost::none,
                              boost::optional<gtsam::Matrix&> J2 = boost::none) const override;

  inline const gtsam::Point3& measured() const
  {
    return measured_;
  }
};

}  // namespace dsc
