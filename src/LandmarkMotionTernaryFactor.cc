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

#include "LandmarkMotionTernaryFactor.h"
#include "GtsamIO.h"

namespace dsc
{

LandmarkMotionTernaryFactor::LandmarkMotionTernaryFactor(gtsam::Key previousPointKey, gtsam::Key currentPointKey,
                                                         gtsam::Key motionKey, const gtsam::Point3& m,
                                                         gtsam::SharedNoiseModel model)
  : gtsam::NoiseModelFactor3<gtsam::Point3, gtsam::Point3, gtsam::Pose3>(model, previousPointKey, currentPointKey,
                                                                         motionKey)
  , measurement_(m)
{
}

gtsam::Vector LandmarkMotionTernaryFactor::evaluateError(const gtsam::Point3& previousPoint,
                                                         const gtsam::Point3& currentPoint, const gtsam::Pose3& H,
                                                         boost::optional<gtsam::Matrix&> J1,
                                                         boost::optional<gtsam::Matrix&> J2,
                                                         boost::optional<gtsam::Matrix&> J3) const
{
  // gtsam::Matrix H2, H3;
  // // gtsam::Vector l2H = H.transformTo(currentPoint, H3, H2);
  // gtsam::Vector l2H = H.transformFrom(previousPoint, H3, H2);
  gtsam::Vector3 l2H = H.inverse() * currentPoint;
  gtsam::Vector3 expected = previousPoint - l2H;
  // gtsam::Vector3 expected = previousPoint - H.inverse() * currentPoint;

  if (J1)
  {
    *J1 = (gtsam::Matrix33() << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0).finished();
    // *J1 = gtsam::numericalDerivative31<gtsam::Vector3, gtsam::Point3, gtsam::Point3, gtsam::Pose3>(
    //   std::bind(&LandmarkMotionTernaryFactor::evaluateError, this, std::placeholders::_1,  std::placeholders::_2,
    //   std::placeholders::_3, boost::none, boost::none, boost::none), previousPoint, currentPoint, H, 1e-3);
  }
  // *J1 = (gtsam::Matrix33() << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0).finished();

  if (J2)
  {
    *J2 = -H.inverse().rotation().matrix();
    // *J2 = (gtsam::Matrix33() << -H2).finished();
    // *J2 = gtsam::numericalDerivative32<gtsam::Vector3, gtsam::Point3, gtsam::Point3, gtsam::Pose3>(
    //   std::bind(&LandmarkMotionTernaryFactor::evaluateError, this, std::placeholders::_1,  std::placeholders::_2,
    //   std::placeholders::_3, boost::none, boost::none, boost::none), previousPoint, currentPoint, H, 1e-3);
  }

  if (J3)
  {
    // *J3 = (gtsam::Matrix36() << -H3).finished();
    // *J3 = gtsam::numericalDerivative33<gtsam::Vector3, gtsam::Point3, gtsam::Point3, gtsam::Pose3>(
    //   std::bind(&LandmarkMotionTernaryFactor::evaluateError, this, std::placeholders::_1,  std::placeholders::_2,
    //   std::placeholders::_3, boost::none, boost::none, boost::none), previousPoint, currentPoint, H, 1e-3);

    Eigen::Matrix<double, 3, 6, Eigen::ColMajor> J;
    J.fill(0);
    // J.block<3, 3>(0, 0) = gtsam::Matrix33::Identity();
    J.block<3, 3>(0, 3) = gtsam::Matrix33::Identity();
    gtsam::Vector3 invHl2 = H.inverse() * currentPoint;
    // J(0, 4) = invHl2(2);
    // J(0, 5) = -invHl2(1);
    // J(1, 3) = -invHl2(2);
    // J(1, 5) = invHl2(0);
    // J(2, 3) = invHl2(1);
    // J(2, 4) = -invHl2(0);
    J(0, 1) = invHl2(2);
    J(0, 2) = -invHl2(1);
    J(1, 0) = -invHl2(2);
    J(1, 2) = invHl2(0);
    J(2, 0) = invHl2(1);
    J(2, 1) = -invHl2(0);

    // Eigen::Matrix<double, 3, 6, Eigen::ColMajor> Jhom = J;
    *J3 = J;
  }

  // return error vector
  return expected;
}

}  // namespace dsc

GTSAM_REGISTER_TYPE(EDGE_SE3_MOTION, dsc::LandmarkMotionTernaryFactor);

namespace gtsam
{
namespace io
{

template <>
bool factor<dsc::LandmarkMotionTernaryFactor>::write(std::ostream& os,
                                                     const dsc::LandmarkMotionTernaryFactor::shared_ptr t)
{
  gtsam::Point3 measurement = t->measurement();
  value<gtsam::Point3>::write(os, measurement);

  // g2o or gtsam...?
  auto gaussianModel = boost::dynamic_pointer_cast<noiseModel::Gaussian>(t->noiseModel());
  gtsam::Matrix Info = gaussianModel->information();
  internal::saveMatrixAsUpperTriangular(os, Info);
  os << t->object_label_ << " ";
  os << t->frame_id_ << " ";
  os << t->camera_pose_key_;
  return os.good();
}

template <>
bool factor<dsc::LandmarkMotionTernaryFactor>::read(std::istream& is, const gtsam::KeyVector& keys,
                                                    gtsam::NonlinearFactor::shared_ptr& factor)
{
  gtsam::Point3 measurement;
  value<gtsam::Point3>::read(is, measurement);
  CHECK_EQ(keys.size(), 3u);

  gtsam::Matrix Info = internal::fullMatrixFromUpperTriangular(is, 3);  // expect a 3 dimensional covariance matrix
  // gtsam::noiseModel::Base::shared_ptr noise_model = gtsam::noiseModel::Gaussian::Information(Info);
  gtsam::noiseModel::Base::shared_ptr noise_model = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);
  auto huber = gtsam::noiseModel::mEstimator::Huber::Create(0.000001,
                                                            gtsam::noiseModel::mEstimator::Base::ReweightScheme::Block);
  gtsam::noiseModel::Base::shared_ptr robust_noise = gtsam::noiseModel::Robust::Create(huber, noise_model);

  // gtsam::PrintKeyVector(keys);
  // LOG(ERROR) << Info;
  // noise_model->print();
  auto factor_cast = boost::make_shared<dsc::LandmarkMotionTernaryFactor>(keys.at(0), keys.at(1), keys.at(2),
                                                                          measurement, robust_noise);
  is >> factor_cast->object_label_;
  is >> factor_cast->frame_id_;
  is >> factor_cast->camera_pose_key_;

  factor = factor_cast;

  return is.good();
}

}  // namespace io
}  // namespace gtsam
