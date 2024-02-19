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

#include "Common.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/base/serialization.h>  //for eigen c++17 workaroud

#include <vector>
#include <boost/serialization/access.hpp>
#include <glog/logging.h>

namespace dsc
{

// https://stackoverflow.com/questions/4633177/c-how-to-wrap-a-float-to-the-interval-pi-pi
//  Floating-point modulo
//  The result (the remainder) has same sign as the divisor.
//  Similar to matlab's mod(); Not similar to fmod() -   Mod(-3,4)= 1   fmod(-3,4)= -3
template <typename T>
T Mod(T x, T y)
{
  static_assert(!std::numeric_limits<T>::is_exact, "Mod: floating-point type expected");

  if (0. == y)
    return x;

  double m = x - y * floor(x / y);

  // handle boundary cases resulted from floating-point cut off:

  if (y > 0)  // modulo range: [0..y)
  {
    if (m >= y)  // Mod(-1e-16             , 360.    ): m= 360.
      return 0;

    if (m < 0)
    {
      if (y + m == y)
        return 0;  // just in case...
      else
        return y + m;  // Mod(106.81415022205296 , _TWO_PI ): m= -1.421e-14
    }
  }
  else  // modulo range: (y..0]
  {
    if (m <= y)  // Mod(1e-16              , -360.   ): m= -360.
      return 0;

    if (m > 0)
    {
      if (y + m == y)
        return 0;  // just in case...
      else
        return y + m;  // Mod(-106.81415022205296, -_TWO_PI): m= 1.421e-14
    }
  }

  return m;
}

// wrap [rad] angle to [0..TWO_PI)
inline double WrapTwoPI(double fAng)
{
  return Mod(fAng, 2.0 * M_PI);
}

struct ErrorPair
{
  double translation = 0;
  double rot = 0;

  ErrorPair()
  {
  }
  ErrorPair(double translation_, double rot_) : translation(translation_), rot(rot_)
  {
  }

  template <class Archive>
  void serialize(Archive& ar, const unsigned int)
  {
    ar& boost::serialization::make_nvp("translation", translation);
    ar& boost::serialization::make_nvp("rot", rot);
  }

  // void average(size_t n) {
  //   if (!utils::fp_equal(translation, 0.0)) {
  //     translation /= static_cast<double>(n);
  //   }

  //   if (!utils::fp_equal(rot, 0.0)) {
  //     rot /= static_cast<double>(n);
  //     rot = world_centric::WrapTwoPI(rot);
  //   }
  // }
};

struct ErrorPairVector : public std::vector<ErrorPair>
{
  // https://en.wikipedia.org/wiki/Circular_mean
  ErrorPair average()
  {
    if (this->empty())
    {
      return ErrorPair();
    }

    ErrorPair error_pair;

    double x = 0, y = 0;
    for (const ErrorPair& pair : *this)
    {
      error_pair.translation += pair.translation;

      y += std::sin(pair.rot);
      x += std::cos(pair.rot);
    }
    const double length = static_cast<double>(this->size());
    x /= length;
    y /= length;
    error_pair.rot = WrapTwoPI(std::atan2(y, x));
    error_pair.translation /= length;

    return error_pair;
  }
};

static inline void calculatePoseError(const gtsam::Pose3& estimated, const gtsam::Pose3& ground_truth, double& t_error,
                                      double& rot_error)
{
  const gtsam::Pose3 pose_change_error = estimated.inverse() * ground_truth;
  const gtsam::Point3& translation_error = pose_change_error.translation();
  // L2 norm - ie. magnitude
  t_error = translation_error.norm();

  const gtsam::Matrix33 rotation_error = pose_change_error.rotation().matrix();
  std::pair<gtsam::Unit3, double> axis_angle = gtsam::Rot3(rotation_error).axisAngle();

  rot_error = axis_angle.second * 180.0 / 3.1415926;
}

// all in world frame
static inline void calculateRelativePoseError(const gtsam::Pose3& previous_pose_estimated,
                                              const gtsam::Pose3& current_pose_estimated,
                                              const gtsam::Pose3& previous_ground_truth,
                                              const gtsam::Pose3& current_ground_truth, double& t_error,
                                              double& rot_error)
{
  // pose change between previous and current
  const gtsam::Pose3 estimated_pose_change = previous_pose_estimated.inverse() * current_pose_estimated;
  const gtsam::Pose3 ground_truth_pose_change = previous_ground_truth.inverse() * current_ground_truth;
  calculatePoseError(estimated_pose_change, ground_truth_pose_change, t_error, rot_error);
}

}  // namespace dsc
