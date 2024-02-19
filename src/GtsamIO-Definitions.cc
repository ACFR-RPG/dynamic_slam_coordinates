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

#include "GtsamIO.h"

#include <iostream>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

namespace gtsam
{
namespace io
{
// clang-format off
GTSAM_REGISTER_TYPE(EDGE_SE3_PRIOR, gtsam::PriorFactor<gtsam::Pose3>);
GTSAM_REGISTER_TYPE(EDGE_SE3:QUAT, gtsam::BetweenFactor<gtsam::Pose3>);
GTSAM_REGISTER_TYPE(VERTEX_TRACKXYZ, gtsam::Point3);
GTSAM_REGISTER_TYPE(VERTEX_SE3:QUAT, gtsam::Pose3);
// clang-format on

template <>
bool value<gtsam::Pose3>::write(std::ostream& os, const gtsam::Pose3& t)
{
  const Point3 p = t.translation();
  const auto q = t.rotation().toQuaternion();
  os << p.x() << " " << p.y() << " " << p.z() << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w();
  return os.good();
}

template <>
bool value<gtsam::Pose3>::read(std::istream& is, gtsam::Pose3& t)
{
  double tx, ty, tz, qx, qy, qz, qw;
  is >> tx >> ty >> tz >> qx >> qy >> qz >> qw;  // TODO: ensure Rot3 is normalized
  t = gtsam::Pose3(gtsam::Rot3(qw, qx, qy, qz), gtsam::Vector3(tx, ty, tz));
  return is.good();
}

template <>
bool value<gtsam::Point3>::write(std::ostream& os, const gtsam::Point3& t)
{
  os << t.x() << " " << t.y() << " " << t.z();
  return os.good();
}

template <>
bool value<gtsam::Point3>::read(std::istream& is, gtsam::Point3& t)
{
  double tx, ty, tz;
  is >> tx >> ty >> tz;
  t = gtsam::Point3(tx, ty, tz);
  return is.good();
}

template <>
bool factor<gtsam::PriorFactor<gtsam::Pose3>>::write(std::ostream& os,
                                                     const gtsam::PriorFactor<gtsam::Pose3>::shared_ptr t)
{
  gtsam::Pose3 measurement = t->prior();
  value<gtsam::Pose3>::write(os, measurement);

  // g2o or gtsam...?
  auto gaussianModel = boost::dynamic_pointer_cast<noiseModel::Gaussian>(t->noiseModel());
  gtsam::Matrix Info = gaussianModel->information();
  internal::saveMatrixAsUpperTriangular(os, Info);
  return os.good();
}

template <>
bool factor<gtsam::PriorFactor<gtsam::Pose3>>::read(std::istream& is, const gtsam::KeyVector& keys,
                                                    gtsam::NonlinearFactor::shared_ptr& factor)
{
  gtsam::Pose3 measurement;
  value<gtsam::Pose3>::read(is, measurement);
  CHECK_EQ(keys.size(), 1u);

  gtsam::Matrix Info = internal::fullMatrixFromUpperTriangular(is, 6);  // expect a 6 dimensional covariance matrix
  gtsam::noiseModel::Base::shared_ptr noise_model = gtsam::noiseModel::Gaussian::Information(Info);

  factor = boost::make_shared<gtsam::PriorFactor<gtsam::Pose3>>(keys.at(0), measurement, noise_model);

  return is.good();
}

template <>
bool factor<gtsam::BetweenFactor<gtsam::Pose3>>::write(std::ostream& os,
                                                       const gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr t)
{
  gtsam::Pose3 measurement = t->measured();
  value<gtsam::Pose3>::write(os, measurement);

  // g2o or gtsam...?
  auto gaussianModel = boost::dynamic_pointer_cast<noiseModel::Gaussian>(t->noiseModel());
  gtsam::Matrix Info = gaussianModel->information();
  internal::saveMatrixAsUpperTriangular(os, Info);
  return os.good();
}

template <>
bool factor<gtsam::BetweenFactor<gtsam::Pose3>>::read(std::istream& is, const gtsam::KeyVector& keys,
                                                      gtsam::NonlinearFactor::shared_ptr& factor)
{
  gtsam::Pose3 measurement;
  value<gtsam::Pose3>::read(is, measurement);
  CHECK_EQ(keys.size(), 2u);

  gtsam::Matrix Info = internal::fullMatrixFromUpperTriangular(is, 6);  // expect a 6 dimensional covariance matrix
  gtsam::noiseModel::Base::shared_ptr noise_model = gtsam::noiseModel::Gaussian::Information(Info);

  // LOG(ERROR) << "\n" << Info;

  factor = boost::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(keys.at(0), keys.at(1), measurement, noise_model);

  return is.good();
}

}  // namespace io
}  // namespace gtsam
