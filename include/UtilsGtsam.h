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

#include <iostream>
#include <string>

#include <glog/logging.h>

#include <Eigen/Core>
#include <type_traits>

#include <opencv2/opencv.hpp>

#include <gtsam/geometry/Unit3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

template <typename T>
struct is_gtsam_factor : std::is_base_of<gtsam::Factor, T>
{
};

template <class T>
static constexpr bool is_gtsam_factor_v = is_gtsam_factor<T>::value;

template <class T>
using enable_if_gtsam_factor = std::enable_if_t<is_gtsam_factor_v<T>>;


//partially taken from: https://github.com/MIT-SPARK/Kimera-VIO/blob/master/include/kimera-vio/utils/UtilsGTSAM.h
namespace dsc
{
namespace utils
{
gtsam::Pose3 cvMatToGtsamPose3(const cv::Mat& H);
// Converts a rotation matrix and translation vector from opencv to gtsam
// pose3
gtsam::Pose3 cvMatsToGtsamPose3(const cv::Mat& R, const cv::Mat& T);

cv::Mat gtsamPose3ToCvMat(const gtsam::Pose3& pose);

/* ------------------------------------------------------------------------ */
// Converts a 3x3 rotation matrix from opencv to gtsam Rot3
gtsam::Rot3 cvMatToGtsamRot3(const cv::Mat& R);

// Converts a 3x1 OpenCV matrix to gtsam Point3
gtsam::Point3 cvMatToGtsamPoint3(const cv::Mat& cv_t);
cv::Mat gtsamPoint3ToCvMat(const gtsam::Point3& point);

template <class T>
static bool getEstimateOfKey(const gtsam::Values& state, const gtsam::Key& key, T* estimate)
{
  if (state.exists(key))
  {
    *CHECK_NOTNULL(estimate) = state.at<T>(key);
    return true;
  }
  else
  {
    return false;
  }
}

}  // namespace utils
}  // namespace dsc
