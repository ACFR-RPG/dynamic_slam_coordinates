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

#include "Point3DFactor.h"
#include "GtsamIO.h"

namespace dsc
{

Point3DFactor::Point3DFactor(gtsam::Key poseKey, gtsam::Key pointKey, const gtsam::Point3& m,
                             gtsam::SharedNoiseModel model)
  : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>(model, poseKey, pointKey), measured_(m.x(), m.y(), m.z())
{
}

// error function
// L is landmark in world coordinates
gtsam::Vector Point3DFactor::evaluateError(const gtsam::Pose3& X, const gtsam::Point3& l,
                                           boost::optional<gtsam::Matrix&> J1, boost::optional<gtsam::Matrix&> J2) const
{
  gtsam::Matrix H1, H2;

  gtsam::Vector expected = X.transformTo(l, H1, H2);

  if (J1)
    *J1 = (gtsam::Matrix36() << H1).finished();

  if (J2)
    *J2 = (gtsam::Matrix33() << H2).finished();

  // return error vector
  return (gtsam::Vector3() << expected - measured_).finished();
}

}  // namespace dsc

GTSAM_REGISTER_TYPE(EDGE_SE3_TRACKXYZ, dsc::Point3DFactor);

namespace gtsam
{
namespace io
{

template <>
bool factor<dsc::Point3DFactor>::write(std::ostream& os, const dsc::Point3DFactor::shared_ptr t)
{
  gtsam::Point3 measurement = t->measured();
  value<gtsam::Point3>::write(os, measurement);

  // g2o or gtsam...?
  auto gaussianModel = boost::dynamic_pointer_cast<noiseModel::Gaussian>(t->noiseModel());
  gtsam::Matrix Info = gaussianModel->information();
  internal::saveMatrixAsUpperTriangular(os, Info);

  os << t->is_dynamic_ << " ";
  return os.good();
}

template <>
bool factor<dsc::Point3DFactor>::read(std::istream& is, const gtsam::KeyVector& keys,
                                      gtsam::NonlinearFactor::shared_ptr& factor)
{
  gtsam::Point3 measurement;
  value<gtsam::Point3>::read(is, measurement);
  CHECK_EQ(keys.size(), 2u);

  gtsam::Matrix Info = internal::fullMatrixFromUpperTriangular(is, 3);  // expect a 3 dimensional covariance matrix
  gtsam::noiseModel::Base::shared_ptr noise_model = gtsam::noiseModel::Gaussian::Information(Info);

  auto huber =
      gtsam::noiseModel::mEstimator::Huber::Create(0.00001, gtsam::noiseModel::mEstimator::Base::ReweightScheme::Block);
  gtsam::noiseModel::Base::shared_ptr robust_noise = gtsam::noiseModel::Robust::Create(huber, noise_model);

  auto factor_cast = boost::make_shared<dsc::Point3DFactor>(keys.at(0), keys.at(1), measurement, robust_noise);

  is >> factor_cast->is_dynamic_;
  factor = factor_cast;

  return is.good();
}

}  // namespace io
}  // namespace gtsam
