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

#include "DataProvider.h"
#include "UtilsGtsam.h"

#include <Eigen/Core>

#include <opencv2/optflow.hpp>
#include <glog/logging.h>
#include <iostream>
#include <iostream>
#include <sys/stat.h>

namespace dsc
{
bool loadRGB(const std::string& image_path, cv::Mat& img)
{
  img = cv::imread(image_path, CV_LOAD_IMAGE_UNCHANGED);
  return true;
}

bool loadDepth(const std::string& image_path, cv::Mat& img)
{
  img = cv::imread(image_path, CV_LOAD_IMAGE_UNCHANGED);
  // For stereo disparity input
  img.convertTo(img, CV_64F);
  // img.convertTo(img, CV_32F);
  return true;
}

bool loadFlow(const std::string& image_path, cv::Mat& img)
{
  img = cv::optflow::readOpticalFlow(image_path);
  return true;
}

bool loadSemanticMask(const std::string& image_path, cv::Mat& mask)
{
  std::ifstream file_mask;
  file_mask.open(image_path.c_str());

  // Main loop
  int count = 0;
  while (!file_mask.eof())
  {
    std::string s;
    getline(file_mask, s);
    if (!s.empty())
    {
      std::stringstream ss;
      ss << s;
      int tmp;
      for (int i = 0; i < mask.cols; ++i)
      {
        ss >> tmp;
        if (tmp != 0)
        {
          mask.at<int>(count, i) = tmp;
        }
        else
        {
          mask.at<int>(count, i) = 0;
        }
      }
      count++;
    }
  }

  file_mask.close();
  return true;
}

void convertToUniqueLabels(const cv::Mat& semantic_instance_mask, cv::Mat& unique_mask)
{
  semantic_instance_mask.copyTo(unique_mask);

  cv::Mat mask_8;
  semantic_instance_mask.convertTo(mask_8, CV_8UC1);
  cv::Mat object_mask = (mask_8 > 0);
  std::vector<cv::Mat> contours;
  cv::Mat hierarchy;
  cv::findContours(object_mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
  cv::Mat drawing = cv::Mat::zeros(semantic_instance_mask.size(), CV_8UC3);
  // LOG(INFO) << "Found unique obj - " <<  contours.size();
  for (size_t i = 0; i < contours.size(); i++)
  {
    cv::Scalar color = cv::Scalar(static_cast<int>(i + 1), 0, 0);
    cv::drawContours(drawing, contours, (int)i, color, CV_FILLED, cv::LINE_8, hierarchy, 0);
  }

  unique_mask = cv::Mat::zeros(semantic_instance_mask.size(), CV_32SC1);
  for (int i = 0; i < unique_mask.rows; i++)
  {
    for (int j = 0; j < unique_mask.cols; j++)
    {
      unique_mask.at<int>(i, j) = drawing.at<cv::Vec3b>(i, j)[0];
    }
  }
}

DataProvider::DataProvider(const std::string& path_to_sequence) : path_to_sequence_(path_to_sequence)
{
  struct stat sb;
  if (!stat(path_to_sequence_.c_str(), &sb) == 0)
  {
    throw std::runtime_error("Path to sequence " + path_to_sequence_ + " is not valid");
  }
}

void DataProvider::load()
{
  loadData(path_to_sequence_);
  LOG(INFO) << "Finished loading sequence " << path_to_sequence_;
}

size_t DataProvider::next(Inputs& input)
{
  return next(input.first, input.second);
}

size_t DataProvider::size() const
{
  return ground_truths_.size();
}

size_t DataProvider::index() const
{
  return index_;
}

size_t DataProvider::next(InputPacket& input_packet, GroundTruthInputPacket::Optional ground_truth)
{
  if (index_ >= this->size())
  {
    return 0;
  }

  cv::Mat rgb, disparity, flow, mask;
  loadImages(index_, rgb, disparity, flow, mask);

  Timestamp timestamp = vTimestamps_[index_];

  input_packet = InputPacket(timestamp, index_, rgb, disparity, flow, mask);

  if (ground_truth)
  {
    *ground_truth = ground_truths_[index_];
  }

  onNext(input_packet, ground_truth);

  index_++;
  return 1;
}

void DataProvider::reset(size_t index)
{
  if (index >= this->size())
  {
    throw std::runtime_error("Index out of bounds");
  }
  index_ = index;
}

bool DataProvider::loadData(const std::string& path_to_sequence)
{
  std::ifstream times_stream;
  std::string strPathTimeFile = path_to_sequence + "/times.txt";
  times_stream.open(strPathTimeFile.c_str());
  while (!times_stream.eof())
  {
    std::string s;
    getline(times_stream, s);
    if (!s.empty())
    {
      std::stringstream ss;
      ss << s;
      double t;
      ss >> t;
      vTimestamps_.push_back(t);
    }
  }
  times_stream.close();

  LOG(INFO) << "N frame - " << vTimestamps_.size();

  // +++ image, depth, semantic and moving object tracking mask +++
  std::string strPrefixImage = path_to_sequence + "/image_0/";      // image  image_0
  std::string strPrefixDepth = path_to_sequence + "/depth/";        // depth_gt  depth  depth_mono_stereo
  std::string strPrefixSemantic = path_to_sequence + "/semantic/";  // semantic_gt  semantic
  std::string strPrefixFlow = path_to_sequence + "/flow/";          // flow_gt  flow

  const int nTimes = vTimestamps_.size();
  vstrFilenamesRGB_.resize(nTimes);
  vstrFilenamesDEP_.resize(nTimes);
  vstrFilenamesSEM_.resize(nTimes);
  vstrFilenamesFLO_.resize(nTimes);

  for (int i = 0; i < nTimes; i++)
  {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << i;
    vstrFilenamesRGB_[i] = strPrefixImage + ss.str() + ".png";
    vstrFilenamesDEP_[i] = strPrefixDepth + ss.str() + ".png";
    vstrFilenamesSEM_[i] = strPrefixSemantic + ss.str() + ".txt";
    vstrFilenamesFLO_[i] = strPrefixFlow + ss.str() + ".flo";
  }

  // +++ ground truth pose +++
  std::string strFilenamePose = path_to_sequence + "/pose_gt.txt";  //  pose_gt.txt  kevin_extrinsics.txt
  // vPoseGT.resize(nTimes);
  std::ifstream fPose;
  fPose.open(strFilenamePose.c_str());
  while (!fPose.eof())
  {
    std::string s;
    getline(fPose, s);
    if (!s.empty())
    {
      std::stringstream ss;
      ss << s;
      int t;
      ss >> t;
      cv::Mat Pose_tmp = cv::Mat::eye(4, 4, CV_64F);

      ss >> Pose_tmp.at<double>(0, 0) >> Pose_tmp.at<double>(0, 1) >> Pose_tmp.at<double>(0, 2) >>
          Pose_tmp.at<double>(0, 3) >> Pose_tmp.at<double>(1, 0) >> Pose_tmp.at<double>(1, 1) >>
          Pose_tmp.at<double>(1, 2) >> Pose_tmp.at<double>(1, 3) >> Pose_tmp.at<double>(2, 0) >>
          Pose_tmp.at<double>(2, 1) >> Pose_tmp.at<double>(2, 2) >> Pose_tmp.at<double>(2, 3) >>
          Pose_tmp.at<double>(3, 0) >> Pose_tmp.at<double>(3, 1) >> Pose_tmp.at<double>(3, 2) >>
          Pose_tmp.at<double>(3, 3);

      // std::vector<double> vec(Pose_tmp.begin<double>(),Pose_tmp.end<double>());
      // vPoseGT_.push_back(parsePose(vec));
      gtsam::Pose3 pose = utils::cvMatToGtsamPose3(Pose_tmp);
      vPoseGT_.push_back(pose);
    }
  }
  fPose.close();

  LOG(INFO) << "Loading object pose";

  // +++ ground truth object pose +++
  std::string strFilenameObjPose = path_to_sequence + "/object_pose.txt";
  std::ifstream fObjPose;
  fObjPose.open(strFilenameObjPose.c_str());

  while (!fObjPose.eof())
  {
    std::string s;
    getline(fObjPose, s);
    if (!s.empty())
    {
      std::stringstream ss;
      ss << s;

      std::vector<double> ObjPose_tmp(10, 0);
      ss >> ObjPose_tmp[0] >> ObjPose_tmp[1] >> ObjPose_tmp[2] >> ObjPose_tmp[3] >> ObjPose_tmp[4] >> ObjPose_tmp[5] >>
          ObjPose_tmp[6] >> ObjPose_tmp[7] >> ObjPose_tmp[8] >> ObjPose_tmp[9];

      ObjectPoseGT object_pose = this->parseObjectGT(ObjPose_tmp);
      vObjPoseGT_.push_back(object_pose);
    }
  }
  fObjPose.close();

  // organise gt poses into vector of arrays
  VectorsXi vObjPoseID(vstrFilenamesRGB_.size());
  for (size_t i = 0; i < vObjPoseGT_.size(); ++i)
  {
    size_t f_id = vObjPoseGT_[i].frame_id;
    if (f_id >= vstrFilenamesRGB_.size())
    {
      break;
    }
    vObjPoseID[f_id].push_back(i);
  }

  // now read image image and add grount truths
  for (size_t frame_id = 0; frame_id < nTimes - 1; frame_id++)
  {
    Timestamp timestamp = vTimestamps_[frame_id];
    GroundTruthInputPacket gt_packet;
    gt_packet.timestamp = timestamp;
    gt_packet.frame_id = frame_id;
    // add ground truths for this fid
    for (int i = 0; i < vObjPoseID[frame_id].size(); i++)
    {
      gt_packet.obj_poses.push_back(vObjPoseGT_[vObjPoseID[frame_id][i]]);
      // sanity check
      CHECK_EQ(gt_packet.obj_poses[i].frame_id, frame_id);
    }
    gt_packet.X_wc = vPoseGT_[frame_id];
    ground_truths_.push_back(gt_packet);
  }

  return true;
}

void DataProvider::loadImages(size_t frame_id, cv::Mat& rgb, cv::Mat& disparity, cv::Mat& flow, cv::Mat& mask) const
{
  loadRGB(vstrFilenamesRGB_[frame_id], rgb);
  loadDepth(vstrFilenamesDEP_[frame_id], disparity);
  loadFlow(vstrFilenamesFLO_[frame_id], flow);

  mask = cv::Mat(rgb.rows, rgb.cols, CV_32SC1);
  loadSemanticMask(vstrFilenamesSEM_[frame_id], mask);
  // convertToUniqueLabels(sem, sem);

  CHECK(!rgb.empty());
  CHECK(!disparity.empty());
  CHECK(!flow.empty());
  CHECK(!mask.empty());
}

KittiSequenceDataProvider::KittiSequenceDataProvider(const std::string& path_to_sequence)
  : DataProvider(path_to_sequence)
{
}

gtsam::Pose3 KittiSequenceDataProvider::parsePose(const std::vector<double>& data) const
{
  // CHECK(data.size() == 16);
  // std::vector<double> data_copy(data);
  // double* data_ptr = data_copy.data();
  // gtsam::Matrix44 T = Eigen::Map<gtsam::Matrix44>(data_ptr, 16);
  //   // gtsam::Matrix44 T = v;
  // // gtsam::Matrix44 T = v.res
  // return gtsam::Pose3(T);
  return gtsam::Pose3();
}

ObjectPoseGT KittiSequenceDataProvider::parseObjectGT(const std::vector<double>& obj_pose_gt) const
{
  // FrameID ObjectID B1 B2 B3 B4 t1 t2 t3 r1
  // Where ti are the coefficients of 3D object location **t** in camera coordinates, and r1 is the Rotation around
  // Y-axis in camera coordinates.
  // B1-4 is 2D bounding box of object in the image, used for visualization.
  CHECK(obj_pose_gt.size() == 10);
  ObjectPoseGT object_pose;
  object_pose.frame_id = obj_pose_gt[0];
  object_pose.object_id = obj_pose_gt[1];
  object_pose.bounding_box.b1 = obj_pose_gt[2];
  object_pose.bounding_box.b2 = obj_pose_gt[3];
  object_pose.bounding_box.b3 = obj_pose_gt[4];
  object_pose.bounding_box.b4 = obj_pose_gt[5];

  cv::Mat t(3, 1, CV_64FC1);
  t.at<double>(0) = obj_pose_gt[6];
  t.at<double>(1) = obj_pose_gt[7];
  t.at<double>(2) = obj_pose_gt[8];

  // from Euler to Rotation Matrix
  cv::Mat R(3, 3, CV_64FC1);

  // assign r vector
  double y = obj_pose_gt[9] + (3.1415926 / 2);  // +(3.1415926/2)
  double x = 0.0;
  double z = 0.0;

  // the angles are in radians.
  double cy = cos(y);
  double sy = sin(y);
  double cx = cos(x);
  double sx = sin(x);
  double cz = cos(z);
  double sz = sin(z);

  double m00, m01, m02, m10, m11, m12, m20, m21, m22;

  m00 = cy * cz + sy * sx * sz;
  m01 = -cy * sz + sy * sx * cz;
  m02 = sy * cx;
  m10 = cx * sz;
  m11 = cx * cz;
  m12 = -sx;
  m20 = -sy * cz + cy * sx * sz;
  m21 = sy * sz + cy * sx * cz;
  m22 = cy * cx;

  R.at<double>(0, 0) = m00;
  R.at<double>(0, 1) = m01;
  R.at<double>(0, 2) = m02;
  R.at<double>(1, 0) = m10;
  R.at<double>(1, 1) = m11;
  R.at<double>(1, 2) = m12;
  R.at<double>(2, 0) = m20;
  R.at<double>(2, 1) = m21;
  R.at<double>(2, 2) = m22;

  // construct 4x4 transformation matrix
  cv::Mat Pose = cv::Mat::eye(4, 4, CV_64F);
  Pose.at<double>(0, 0) = R.at<double>(0, 0);
  Pose.at<double>(0, 1) = R.at<double>(0, 1);
  Pose.at<double>(0, 2) = R.at<double>(0, 2);
  Pose.at<double>(0, 3) = t.at<double>(0);
  Pose.at<double>(1, 0) = R.at<double>(1, 0);
  Pose.at<double>(1, 1) = R.at<double>(1, 1);
  Pose.at<double>(1, 2) = R.at<double>(1, 2);
  Pose.at<double>(1, 3) = t.at<double>(1);
  Pose.at<double>(2, 0) = R.at<double>(2, 0);
  Pose.at<double>(2, 1) = R.at<double>(2, 1);
  Pose.at<double>(2, 2) = R.at<double>(2, 2);
  Pose.at<double>(2, 3) = t.at<double>(2);

  object_pose.pose = utils::cvMatToGtsamPose3(Pose);
  return object_pose;
}

OmdSequenceDataProvider::OmdSequenceDataProvider(const std::string& path_to_sequence) : DataProvider(path_to_sequence)
{
}

ObjectPoseGT OmdSequenceDataProvider::parseObjectGT(const std::vector<double>& data) const
{
  ObjectPoseGT object_pose;
  object_pose.frame_id = static_cast<size_t>(data[0]);
  object_pose.object_id = static_cast<size_t>(data[1]);

  const auto frame_id = object_pose.frame_id;

  // omd does not have bounding box information so we need to find them ourselves... so gross!!
  // this will happen in the onNext function callback so we dont have to load all the images twice!!

  // assign t vector
  cv::Mat t(3, 1, CV_64F);
  t.at<double>(0) = data[2];
  t.at<double>(1) = data[3];
  t.at<double>(2) = data[4];

  // from axis-angle to Rotation Matrix
  cv::Mat R(3, 3, CV_64F);
  cv::Mat Rvec(3, 1, CV_64F);

  // assign r vector
  Rvec.at<double>(0, 0) = data[5];
  Rvec.at<double>(0, 1) = data[6];
  Rvec.at<double>(0, 2) = data[7];

  // *******************************************************************

  const double angle = std::sqrt(data[5] * data[5] + data[6] * data[6] + data[7] * data[7]);

  if (angle > 0)
  {
    Rvec.at<double>(0, 0) = Rvec.at<double>(0, 0) / angle;
    Rvec.at<double>(0, 1) = Rvec.at<double>(0, 1) / angle;
    Rvec.at<double>(0, 2) = Rvec.at<double>(0, 2) / angle;
  }

  const double s = std::sin(angle);
  const double c = std::cos(angle);

  const double v = 1 - c;
  const double x = Rvec.at<double>(0, 0);
  const double y = Rvec.at<double>(0, 1);
  const double z = Rvec.at<double>(0, 2);
  const double xyv = x * y * v;
  const double yzv = y * z * v;
  const double xzv = x * z * v;

  R.at<double>(0, 0) = x * x * v + c;
  R.at<double>(0, 1) = xyv - z * s;
  R.at<double>(0, 2) = xzv + y * s;
  R.at<double>(1, 0) = xyv + z * s;
  R.at<double>(1, 1) = y * y * v + c;
  R.at<double>(1, 2) = yzv - x * s;
  R.at<double>(2, 0) = xzv - y * s;
  R.at<double>(2, 1) = yzv + x * s;
  R.at<double>(2, 2) = z * z * v + c;

  // construct 4x4 transformation matrix
  cv::Mat Pose = cv::Mat::eye(4, 4, CV_64F);
  Pose.at<double>(0, 0) = R.at<double>(0, 0);
  Pose.at<double>(0, 1) = R.at<double>(0, 1);
  Pose.at<double>(0, 2) = R.at<double>(0, 2);
  Pose.at<double>(0, 3) = t.at<double>(0);
  Pose.at<double>(1, 0) = R.at<double>(1, 0);
  Pose.at<double>(1, 1) = R.at<double>(1, 1);
  Pose.at<double>(1, 2) = R.at<double>(1, 2);
  Pose.at<double>(1, 3) = t.at<double>(1);
  Pose.at<double>(2, 0) = R.at<double>(2, 0);
  Pose.at<double>(2, 1) = R.at<double>(2, 1);
  Pose.at<double>(2, 2) = R.at<double>(2, 2);
  Pose.at<double>(2, 3) = t.at<double>(2);

  object_pose.pose = utils::cvMatToGtsamPose3(Pose);

  return object_pose;
}

void OmdSequenceDataProvider::onNext(InputPacket& input_packet, GroundTruthInputPacket::Optional ground_truth)
{
  if (!ground_truth)
  {
    return;
  }

  GroundTruthInputPacket& gt_packet = ground_truth.get();
  const cv::Mat& semantic_mask = input_packet.images.semantic_mask;
  CHECK(!semantic_mask.empty());
  CHECK(input_packet.frame_id == gt_packet.frame_id);

  for (ObjectPoseGT& gt_object_pose : gt_packet.obj_poses)
  {
    upateGTBoundingBox(gt_object_pose, semantic_mask);
  }
}

void OmdSequenceDataProvider::upateGTBoundingBox(ObjectPoseGT& gt_object_pose, const cv::Mat& semantic_mask)
{
  const int object_id = gt_object_pose.object_id;
  cv::Mat mask = semantic_mask == object_id;

  std::vector<std::vector<cv::Point> > contours;
  cv::findContours(mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

  std::vector<std::vector<cv::Point> > contours_poly(contours.size());
  std::vector<cv::Rect> boundRect(contours.size());
  cv::Mat drawing = cv::Mat::zeros(mask.size(), CV_8UC3);

  cv::Rect* largest_bounding_box = nullptr;
  size_t largest_area = 0;

  static cv::RNG rng(12345);
  for (size_t i = 0; i < contours.size(); i++)
  {
    //  drawContours( drawing, contours_poly, (int)i, color );
    cv::approxPolyDP(contours[i], contours_poly[i], 3, true);

    boundRect[i] = boundingRect(contours_poly[i]);

    const auto area = boundRect[i].area();
    if (area > largest_area)
    {
      largest_area = area;
      largest_bounding_box = &boundRect[i];
    }
  }

  if (largest_bounding_box != nullptr)
  {
    // cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

    const cv::Rect& bb = *largest_bounding_box;

    // we need to do some gross conversions from opencv
    // we want the bottom left(x1,y1) and top right (x2, y2)

    // int x1 = bb.tl().x;
    // int y2 = bb.tl().y;

    // int x2 = bb.br().x;
    // int y1 = bb.br().y;

    // stickign to open CV conventions where image starts top right
    int x1 = bb.tl().x;
    int y1 = bb.tl().y;

    int x2 = bb.br().x;
    int y2 = bb.br().y;
    gt_object_pose.bounding_box.b1 = x1;
    gt_object_pose.bounding_box.b2 = y1;
    gt_object_pose.bounding_box.b3 = x2;
    gt_object_pose.bounding_box.b4 = y2;
  }
}

}  // namespace dsc
