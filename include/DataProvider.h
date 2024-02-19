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

#include "Types.h"
#include <utility>  //for pair

namespace dsc
{
bool loadRGB(const std::string& image_path, cv::Mat& img);
bool loadDepth(const std::string& image_path, cv::Mat& img);
bool loadFlow(const std::string& image_path, cv::Mat& img);

// mask must be resized before hand
bool loadSemanticMask(const std::string& image_path, cv::Mat& mask);
void convertToUniqueLabels(const cv::Mat& semantic_instance_mask, cv::Mat& unique_mask);

class DataProvider
{
public:
  DSC_POINTER_TYPEDEFS(DataProvider);

  using Inputs = std::pair<InputPacket, GroundTruthInputPacket>;
  using InputsVector = std::vector<Inputs>;

  DataProvider(const std::string& path_to_sequence);
  virtual ~DataProvider() = default;

  void load();

  size_t next(Inputs& input);
  size_t next(InputPacket& input_packet, GroundTruthInputPacket::Optional ground_truth);
  size_t size() const;

  size_t index() const;

  // resets index to start or i -> throws exception if i out of bounds
  void reset(size_t index = 0);

protected:
  const std::string path_to_sequence_;

  virtual gtsam::Pose3 parsePose(const std::vector<double>& data) const = 0;
  virtual ObjectPoseGT parseObjectGT(const std::vector<double>& data) const = 0;

  // both are modifable
  virtual void onNext(InputPacket& input_packet, GroundTruthInputPacket::Optional ground_truth)
  {
  }

protected:
  /**
   * @brief Preloads all the filename vectors as well as the ground truth object and camera poses as well as
   * timestamps.
   *
   * @param path_to_sequence
   * @return true
   * @return false
   */
  bool loadData(const std::string& path_to_sequence);
  void loadImages(size_t frame_id, cv::Mat& rgb, cv::Mat& disparity, cv::Mat& flow, cv::Mat& mask) const;

  std::vector<std::string> vstrFilenamesRGB_;
  std::vector<std::string> vstrFilenamesDEP_;
  std::vector<std::string> vstrFilenamesSEM_;
  std::vector<std::string> vstrFilenamesFLO_;
  std::vector<gtsam::Pose3> vPoseGT_;
  std::vector<ObjectPoseGT> vObjPoseGT_;
  std::vector<Timestamp> vTimestamps_;

  // per frame ground truth
  std::vector<GroundTruthInputPacket> ground_truths_;

private:
  // should correspond with the frame_id of the data
  size_t index_ = 0;
};

class KittiSequenceDataProvider : public DataProvider
{
public:
  KittiSequenceDataProvider(const std::string& path_to_sequence);

  // size_t next(InputPacket& input_packet, GroundTruthInputPacket::Optional ground_truth) override;
  // size_t size() const override;

private:
  gtsam::Pose3 parsePose(const std::vector<double>& data) const override;
  ObjectPoseGT parseObjectGT(const std::vector<double>& data) const override;
};

class OmdSequenceDataProvider : public DataProvider
{
public:
  OmdSequenceDataProvider(const std::string& path_to_sequence);

private:
  gtsam::Pose3 parsePose(const std::vector<double>& data) const override
  {
    return gtsam::Pose3();
  }
  ObjectPoseGT parseObjectGT(const std::vector<double>& data) const override;
  void onNext(InputPacket& input_packet, GroundTruthInputPacket::Optional ground_truth) override;

  void upateGTBoundingBox(ObjectPoseGT& gt_object_pose, const cv::Mat& semantic_mask);
};

}  // namespace dsc
