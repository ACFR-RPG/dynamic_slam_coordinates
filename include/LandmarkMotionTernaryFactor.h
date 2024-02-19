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
#include <gtsam/base/numericalDerivative.h>

// A landmark motion ternary factor
// The factor contains the difference between l_{k+1} and the motion vertex _k H_{k+1} applied to l_k
// The error vector will be zeros(3,1) - [l_k - _k H_{k+1}^-1 l_{k+1}]'
namespace dsc
{
class LandmarkMotionTernaryFactor : public gtsam::NoiseModelFactor3<gtsam::Point3, gtsam::Point3, gtsam::Pose3>
{
private:
  // measurement information
  gtsam::Vector3 measurement_;

public:
  typedef boost::shared_ptr<LandmarkMotionTernaryFactor> shared_ptr;
  typedef LandmarkMotionTernaryFactor This;
  int object_label_;
  int frame_id_;
  gtsam::Key camera_pose_key_;  // of current point (not previous)
  /**
   * Constructor
   * @param motion     pose transformation (motion)
   * @param model      noise model for ternary factor
   * @param m          Vector measurement
   */
  LandmarkMotionTernaryFactor(gtsam::Key previousPointKey, gtsam::Key currentPointKey, gtsam::Key motionKey,
                              const gtsam::Point3& m, gtsam::SharedNoiseModel model);

  gtsam::NonlinearFactor::shared_ptr clone() const override
  {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  // error function
  // l1 is the previouus point
  gtsam::Vector evaluateError(const gtsam::Point3& previousPoint, const gtsam::Point3& currentPoint,
                              const gtsam::Pose3& H, boost::optional<gtsam::Matrix&> J1 = boost::none,
                              boost::optional<gtsam::Matrix&> J2 = boost::none,
                              boost::optional<gtsam::Matrix&> J3 = boost::none) const override;

  inline gtsam::Key PreviousPointKey() const
  {
    return key1();
  }
  inline gtsam::Key CurrentPointKey() const
  {
    return key2();
  }
  inline gtsam::Key MotionKey() const
  {
    return key3();
  }

  inline gtsam::Vector3 measurement() const
  {
    return measurement_;
  }

  inline int objectLabel() const
  {
    return object_label_;
  }

  inline int frameId() const
  {
    return frame_id_;
  }
};

}  // namespace dsc
