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

#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <gtsam/geometry/Pose3.h>
#include <string>

namespace dsc
{
namespace utils
{
void DrawCircleInPlace(cv::Mat& img, const cv::Point2d& point, const cv::Scalar& colour, const double msize = 0.4);

#define CHECK_MAT_TYPES(mat1, mat2)                                                                                    \
  using namespace VDO_SLAM::utils;                                                                                     \
  CHECK_EQ(mat1.type(), mat2.type()) << "Matricies should be of the same type ( " << cvTypeToString(mat1.type())       \
                                     << " vs. " << cvTypeToString(mat2.type()) << ")."

std::string cvTypeToString(int type);
std::string cvTypeToString(const cv::Mat& mat);

cv::Mat concatenateImagesHorizontally(const cv::Mat& left_img, const cv::Mat& right_img);

cv::Mat concatenateImagesVertically(const cv::Mat& top_img, const cv::Mat& bottom_img);

cv::Affine3d matPoseToCvAffine3d(const cv::Mat& pose);

// should be in utils GTSAM but meh
cv::Affine3d gtsamPose3ToCvAffine3d(const gtsam::Pose3& pose);

// // applies a rotation to a pose in the camera frame (z forward)
// // to get it to align with the standard axis
cv::Mat alignCameraPoseToWorld(const cv::Mat& pose);

void flowToRgb(const cv::Mat& flow, cv::Mat& rgb);

cv::Scalar getObjectColour(int label);

}  // namespace utils
}  // namespace dsc
