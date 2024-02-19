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

/**
 * MOTIONFACTOR must constrain at least two points with the previous being key1 and the current being key2
 * and a motion (pose3) with the function MotionKey. ALso must have a variable camera_pose_key_ which indicates the
 * key of the pose the second (current) point was seen from
 * - TODO: clean up
 */

#include "Types.h"
#include "UtilsGtsam.h"
#include "GtsamIO.h"
#include "Camera.h"
#include "Metrics.h"
#include "DataProvider.h"
#include "UtilsOpenCV.h"

#include "Optimizer.h"
#include "ObjectCentric.h"
#include "Common.h"

#include "Point3DFactor.h"
#include "LandmarkMotionTernaryFactor.h"

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/inference/Key.h>
#include <gtsam/inference/inferenceExceptions.h>
#include <opencv2/opencv.hpp>

DECLARE_string(kitti_dataset);
DECLARE_int32(input_type);  // 0 = kitti, 1 = OMD, 2 = sim

/**
 * @brief Get the output path (as set by gflags) as a valid path and checks if the folder exists
 * 
 * @return std::string 
 */
std::string getOutputPath();

template <typename FUNCTION>
class OptimizationOutputWriter
{
public:
  OptimizationOutputWriter(const std::string output_file) : output_file_(output_file)
  {
    os_.open(output_file, std::ios::out);
  }

  virtual ~OptimizationOutputWriter()
  {
    if (os_.is_open())
      os_.close();
  }

  virtual FUNCTION getHook() = 0;

protected:
  std::string output_file_;
  std::ofstream os_;
};

class IterationOutputWriter : public OptimizationOutputWriter<gtsam::NonlinearOptimizerParams::IterationHook>
{
public:
  IterationOutputWriter(const std::string output_file) : OptimizationOutputWriter(output_file)
  {
    os_ << "iteration, error_before, error_after\n";
  }

  gtsam::NonlinearOptimizerParams::IterationHook getHook()
  {
    return std::bind(&IterationOutputWriter::iterationHook, this, std::placeholders::_1, std::placeholders::_2,
                     std::placeholders::_3);
  }

private:
  void iterationHook(size_t iteration, double error_before, double error_after)
  {
    os_ << std::setprecision(12) << iteration << "," << error_before << "," << error_after << '\n';
  }
};

class PerInnerIterationOutputWriter
  : public OptimizationOutputWriter<gtsam::LevenbergMarquardtWithLoggingParams::InnerIterationHook>
{
public:
  PerInnerIterationOutputWriter(const std::string output_file) : OptimizationOutputWriter(output_file)
  {
    os_ << "iteration, inner_iterations, new_error, linear_costchange, nonlinear_costchange, lambda\n";
  }

  gtsam::LevenbergMarquardtWithLoggingParams::InnerIterationHook getHook()
  {
    return std::bind(&PerInnerIterationOutputWriter::iterationHook, this, std::placeholders::_1);
  }

private:
  void iterationHook(const gtsam::LMLoggingInfo& logging_info)
  {
    os_ << std::setprecision(12) << logging_info.iterations << "," << logging_info.inner_iterations << ","
        << logging_info.error << "," << logging_info.linear_costchange << "," << logging_info.non_linear_costchange
        << "," << logging_info.lambda << '\n';
  }
};

template <typename MOTIONFACTOR = dsc::LandmarkMotionTernaryFactor>
GraphDeconstruction deconstructGraph(const gtsam::NonlinearFactorGraph& graph)
{
  // mapping of key to tracklet
  size_t max_tracklet = 0;
  std::map<gtsam::Key, size_t> key_to_tracklet_id;
  std::map<gtsam::Key, Feature::shared_ptr> key_to_feature;
  LOG(INFO) << "Deconstructing graph of size " << graph.size();

  GraphDeconstruction graph_deconstruction;

  for (size_t i = 0; i < graph.size(); i++)
  {
    const gtsam::NonlinearFactor::shared_ptr& factor = graph.at(i);
    const gtsam::KeyVector& keys = factor->keys();

    const auto& landmark_tenrary_factor = boost::dynamic_pointer_cast<MOTIONFACTOR>(factor);
    if (landmark_tenrary_factor)
    {
      graph_deconstruction.landmark_ternary_factor_idx.push_back(i);
      graph_deconstruction.camera_pose_keys.insert(landmark_tenrary_factor->camera_pose_key_);
      graph_deconstruction.object_motion_keys.insert(landmark_tenrary_factor->MotionKey());

      gtsam::Key previous_point = landmark_tenrary_factor->key1();
      gtsam::Key current_point = landmark_tenrary_factor->key2();

      CHECK(key_to_tracklet_id.find(current_point) == key_to_tracklet_id.end())
          << " Current key " << current_point
          << " is in key to tracklet map but shouldnt be"
             "prev key "
          << previous_point << " motion " << landmark_tenrary_factor->MotionKey();

      // if previous point is in graph then use the associated tracklet to label and associte the current point
      if (key_to_tracklet_id.find(previous_point) != key_to_tracklet_id.end())
      {
        size_t tracklet_id = key_to_tracklet_id.at(previous_point);
        key_to_tracklet_id[current_point] = tracklet_id;

        // update previous point to the current motion key
        DynamicFeature::shared_ptr previous_feature =
            std::dynamic_pointer_cast<DynamicFeature>(key_to_feature[previous_point]);
        CHECK(previous_feature);
        CHECK(!previous_feature->motion_key.is_initialized());
        previous_feature->motion_key = landmark_tenrary_factor->MotionKey();

        DynamicFeature::shared_ptr feature = std::make_shared<DynamicFeature>();
        feature->key = current_point;
        feature->is_dynamic_ = true;
        // feature->motion_key = landmark_tenrary_factor->MotionKey();
        feature->tracklet_id = tracklet_id;
        key_to_feature[current_point] = feature;
      }
      else
      {
        max_tracklet++;
        key_to_tracklet_id[previous_point] = max_tracklet;
        key_to_tracklet_id[current_point] = max_tracklet;

        {
          DynamicFeature::shared_ptr feature = std::make_shared<DynamicFeature>();
          feature->is_dynamic_ = true;
          feature->key = previous_point;
          feature->motion_key = landmark_tenrary_factor->MotionKey();  // this is correct as we want the motion to
                                                                       // propogate forward from the previous point
          feature->tracklet_id = max_tracklet;
          key_to_feature[previous_point] = feature;
        }

        {
          DynamicFeature::shared_ptr feature = std::make_shared<DynamicFeature>();
          feature->is_dynamic_ = true;
          feature->key = current_point;
          // feature->motion_key = landmark_tenrary_factor->MotionKey();
          feature->tracklet_id = max_tracklet;
          key_to_feature[current_point] = feature;
        }
      }
    }

    const auto& point3d_factor = boost::dynamic_pointer_cast<dsc::Point3DFactor>(factor);
    if (point3d_factor)
    {
      gtsam::Key point_key = point3d_factor->key2();

      if (point3d_factor->is_dynamic_)
      {
        graph_deconstruction.dynamic_3d_point_idx.push_back(i);
      }
      else
      {
        graph_deconstruction.static_3d_point_idx.push_back(i);
        graph_deconstruction.camera_pose_keys.insert(point3d_factor->key1());

        StaticFeature::shared_ptr feature;
        // all static points that are tracket should have the same key in the graph
        if (key_to_tracklet_id.find(point_key) == key_to_tracklet_id.end())
        {
          max_tracklet++;
          key_to_tracklet_id[point_key] = max_tracklet;
          feature = std::make_shared<StaticFeature>();
          feature->key = point_key;
          feature->tracklet_id = max_tracklet;
          feature->camera_pose_keys.push_back(point3d_factor->key1());
          feature->measurements.push_back(point3d_factor->measured());
          // add feature for the static points since we have never seen one before
          key_to_feature[point_key] = feature;
        }
        else
        {
          feature = std::dynamic_pointer_cast<StaticFeature>(key_to_feature[point_key]);
          CHECK(feature);
          CHECK_EQ(feature->tracklet_id, key_to_tracklet_id.at(point_key));
          feature->measurements.push_back(point3d_factor->measured());
          feature->camera_pose_keys.push_back(point3d_factor->key1());
        }
      }
    }
  }

  LOG(INFO) << "Parsing dynamic Point3DFactor";
  // update all the dynamic points from the Point3DFactor since it was too hard to do in the same loop as above
  for (size_t i = 0; i < graph.size(); i++)
  {
    const gtsam::NonlinearFactor::shared_ptr& factor = graph.at(i);
    const auto& point3d_factor = boost::dynamic_pointer_cast<dsc::Point3DFactor>(factor);
    if (point3d_factor && point3d_factor->is_dynamic_)
    {
      gtsam::Key point_key = point3d_factor->key2();
      // CHECK(key_to_tracklet_id.find(point_key) != key_to_tracklet_id.end()) << " point key " << point_key;

      if (key_to_tracklet_id.find(point_key) == key_to_tracklet_id.end())
      {
        // ive' noticed this get triggered on some datasets (eg KITTI 0020) when points which are classified as dynamic
        // do not appear in a motion factor - and we iterate over the motion factors to find the dynamic tracklets
        // I think this might be a very small bug in the original vdo-slam implementation
        // LOG(WARNING) << "Missing point key " << point_key << " at graph idx " << i;
        key_to_feature.erase(point_key);
        key_to_tracklet_id.erase(point_key);
        continue;
      }
      Feature::shared_ptr feature = key_to_feature.at(point_key);

      auto dynamic_feature = std::dynamic_pointer_cast<DynamicFeature>(key_to_feature.at(point_key));
      CHECK(dynamic_feature);
      dynamic_feature->camera_pose_key = point3d_factor->key1();
      dynamic_feature->measurement = point3d_factor->measured();
      CHECK(feature->is_dynamic_);  // cannot use Feature::IsDynamic() yet as we have not updated the
                                    // object_label key (this is done in the construct object motion
                                    // results map)
    }
  }

  LOG(INFO) << "Parsing pose3 between factors";
  // after this go over the between factors
  for (size_t i = 0; i < graph.size(); i++)
  {
    const gtsam::NonlinearFactor::shared_ptr& factor = graph.at(i);
    const gtsam::KeyVector& keys = factor->keys();

    const auto& between_factor = boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(factor);

    if (between_factor)
    {
      const gtsam::Key from_key = keys.at(0);
      const gtsam::Key to_key = keys.at(1);

      // in camera poses
      bool from_key_is_in =
          graph_deconstruction.camera_pose_keys.find(from_key) != graph_deconstruction.camera_pose_keys.end();
      bool to_key_is_in =
          graph_deconstruction.camera_pose_keys.find(to_key) != graph_deconstruction.camera_pose_keys.end();

      if (from_key_is_in && to_key_is_in)
        graph_deconstruction.between_factor_idx.push_back(i);

      // in object_motion poses
      from_key_is_in =
          graph_deconstruction.object_motion_keys.find(from_key) != graph_deconstruction.object_motion_keys.end();
      to_key_is_in =
          graph_deconstruction.object_motion_keys.find(to_key) != graph_deconstruction.object_motion_keys.end();

      if (from_key_is_in && to_key_is_in)
        graph_deconstruction.smoothing_factor_idx.push_back(i);
    }
  }

  graph_deconstruction.tracklets = TrackletMap(key_to_tracklet_id, key_to_feature);
  LOG(INFO) << "Finished parsing graph";

  return graph_deconstruction;
}

template <typename MOTIONFACTOR = dsc::LandmarkMotionTernaryFactor>
ObjectLabelResultMap constructObjectMotionResultMap(dsc::DataProvider& dataprovider,
                                                    GraphDeconstruction& graph_deconstruction,
                                                    const gtsam::NonlinearFactorGraph& graph,
                                                    const gtsam::Values& initial_values,
                                                    boost::optional<const gtsam::Values&> optimized_values)
{
  dataprovider.reset();

  gtsam::KeyVector cam_key_vector(graph_deconstruction.camera_pose_keys.size());
  std::copy(graph_deconstruction.camera_pose_keys.begin(), graph_deconstruction.camera_pose_keys.end(),
            cam_key_vector.begin());

  // set should be ordered but lets just guranatee?
  // first key should correspond to the first frame (which is frame 0 in the dataset)
  std::sort(cam_key_vector.begin(), cam_key_vector.end());

  std::shared_ptr<dsc::CameraParams> camera_params;

  // kitti
  if (FLAGS_input_type == 0)
  {
    const int kitti_dataset = std::stoi(FLAGS_kitti_dataset);
    LOG(INFO) << "dataset " << kitti_dataset;
    if (kitti_dataset >= 0 && kitti_dataset <= 13)
    {
      dsc::CameraParams::Intrinsics intrinsics({ 721.5377, 721.5377, 609.5593, 172.8540 });
      dsc::CameraParams::Distortion distortion({ 0, 0, 0, 0 });

      cv::Size image_size(1242, 375);
      camera_params = std::make_shared<dsc::CameraParams>(intrinsics, distortion, image_size, "none", 0);
    }
    else
    {
      dsc::CameraParams::Intrinsics intrinsics({ 718.8560, 718.8560, 607.1928, 185.2157 });
      dsc::CameraParams::Distortion distortion({ 0, 0, 0, 0 });

      cv::Size image_size(1242, 375);
      camera_params = std::make_shared<dsc::CameraParams>(intrinsics, distortion, image_size, "none", 0);
    }
  }
  // OMD
  else if (FLAGS_input_type == 1)
  {
    LOG(INFO) << "Constructing object motion result map with OMD camera params";
    dsc::CameraParams::Intrinsics intrinsics(
        { 974.8409602402362, 975.2969974293134, 637.9512716907054, 482.0369315847473 });
    dsc::CameraParams::Distortion distortion(
        { -0.31811570871103184, 0.10347911678780099, 0.0009600873384515784, 0.0020253178045684227 });

    cv::Size image_size(1280, 960);
    camera_params = std::make_shared<dsc::CameraParams>(intrinsics, distortion, image_size, "none", 0);
  }

  CHECK_NOTNULL(camera_params);
  dsc::Camera camera(*camera_params);

  dsc::InputPacket input;
  dsc::GroundTruthInputPacket gt;
  dsc::GroundTruthInputPacket previous_gt;

  ObjectLabelResultMap object_result_map;
  ObjectFrameIdResultMap frame_id_result_map;

  std::map<int, cv::Mat> debug_images;

  // get ground truth first by iterating over every frame
  for (size_t i = 0; i < cam_key_vector.size(); i++)
  {
    CHECK(dataprovider.next(input, gt));
    if (i > 0)
    {
      for (const dsc::ObjectPoseGT& obj_pose_gt : previous_gt.obj_poses)
      {
        dsc::ObjectPoseGT obj_pose_gt_prev, obj_pose_gt_curr;

        if (!gt.getObjectGTPair(previous_gt, obj_pose_gt.object_id, obj_pose_gt_curr, obj_pose_gt_prev))
        {
          // LOG(WARNING) << "No obj gt for object in previous and current frame";
          continue;
        }
        CHECK(obj_pose_gt_prev.object_id == obj_pose_gt_curr.object_id);
        const int object_id = obj_pose_gt_prev.object_id;

        // for t-1
        gtsam::Pose3 L_c_gt_prev = obj_pose_gt_prev.pose;
        gtsam::Pose3 L_w_gt_prev = previous_gt.X_wc * L_c_gt_prev;

        // for t
        gtsam::Pose3 L_c_gt_curr = obj_pose_gt_curr.pose;
        gtsam::Pose3 L_w_gt_curr = gt.X_wc * L_c_gt_curr;

        // H between t-1 and t in the object frame ground truth
        // gtsam::Pose3 H_L_gt = L_c_gt_prev.inverse() * L_c_gt_curr;
        gtsam::Pose3 H_L_gt = L_w_gt_prev.inverse() * L_w_gt_curr;

        // H between t-1 and t in the camera frame at t-1
        gtsam::Pose3 H_c_gt = L_c_gt_prev * H_L_gt * L_c_gt_prev.inverse();

        // H between t-1 and t in the world frame
        gtsam::Pose3 H_w_gt = L_w_gt_prev * H_L_gt * L_w_gt_prev.inverse();

        gtsam::Pose3 H_w_gt_need_for_speed = L_w_gt_curr * L_w_gt_prev.inverse();
        // CHECK(H_w_gt.equals(H_w_gt_need_for_speed, 0.01)) << H_w_gt << " " << H_w_gt_need_for_speed;

        // use previous frame
        const int frame_id = obj_pose_gt_prev.frame_id;

        // add to ground truth
        if (object_result_map.find(object_id) == object_result_map.end())
        {
          // is a new object
          // LOG(INFO) << "New object label " << object_id;
          object_result_map[object_id] = ObjectMotionResults();
        }

        if (frame_id_result_map.find(frame_id) == frame_id_result_map.end())
        {
          // frame id
          // LOG(INFO) << "New frame id " << frame_id;
          frame_id_result_map[frame_id] = ObjectMotionResults();
        }

        if (debug_images.find(frame_id) == debug_images.end())
        {
          debug_images[frame_id] = cv::Mat::zeros(camera.Params().image_size, CV_8UC3);
        }

        ObjectMotionResult::shared_ptr result = std::make_shared<ObjectMotionResult>();
        result->object_label = object_id;
        CHECK(object_id != 0);
        result->frame_id = frame_id;  // previous frame
        result->object_pose_first_world_gt = L_w_gt_prev;
        result->object_pose_second_world_gt = L_w_gt_curr;
        result->ground_truth_L = H_L_gt;
        result->ground_truth_W = H_w_gt;
        result->ground_truth_C = H_c_gt;
        result->first_frame_camera_pose_gt = previous_gt.X_wc;
        result->second_frame_camera_pose_gt = gt.X_wc;
        result->first_frame_bounding_box = obj_pose_gt_prev.bounding_box;
        result->second_frame_bounding_box = obj_pose_gt_curr.bounding_box;

        {
          ObjectMotionResults& object_motions = object_result_map.at(object_id);
          object_motions.push_back(result);
        }

        {
          ObjectMotionResults& object_motions = frame_id_result_map.at(frame_id);
          object_motions.push_back(result);
        }

        cv::Point pt1(obj_pose_gt_prev.bounding_box.b1, obj_pose_gt_prev.bounding_box.b2);
        cv::Point pt2(obj_pose_gt_prev.bounding_box.b3, obj_pose_gt_prev.bounding_box.b4);
        cv::rectangle(debug_images.at(frame_id), pt1, pt2, cv::Scalar(0, 255, 0), 2);
      }
    }

    previous_gt = gt;
  }

  struct
  {
    bool operator()(dsc::KeypointCV kp, dsc::ObjectPoseGT::BoundingBox bounding_box)
    {
      int x = kp.pt.x;
      int y = kp.pt.y;

      if (x < bounding_box.b1 || x > bounding_box.b3 || y < bounding_box.b2 || y > bounding_box.b4)
      {
        return false;
      }
      else
      {
        return true;
      }
    }

  } isKpInBoundingBox;

  // now iterate over objects
  int total = 0;
  for (size_t idx : graph_deconstruction.landmark_ternary_factor_idx)
  {
    auto factor = graph.at(idx);
    auto landmark_tenrary_factor = boost::dynamic_pointer_cast<MOTIONFACTOR>(factor);
    CHECK(landmark_tenrary_factor);

    int second_frame = landmark_tenrary_factor->frameId();  // in vdo-slam we save the current id
    gtsam::Key second_camera_pose_key =
        landmark_tenrary_factor->camera_pose_key_;  // we also save the second camera pose key
    // check these match up
    int second_frame_idx = second_frame;  // frame id's start at 0
    CHECK_EQ(cam_key_vector[second_frame_idx], second_camera_pose_key) << " Frame id = " << second_frame_idx;
    int first_frame_idx = second_frame_idx - 1;  // assume frames are consequative?
    CHECK(first_frame_idx >= 0);
    gtsam::Key first_camera_pose_key = cam_key_vector[first_frame_idx];

    gtsam::Pose3 first_frame_camera_pose = initial_values.at<gtsam::Pose3>(first_camera_pose_key);
    gtsam::Pose3 second_frame_camera_pose = initial_values.at<gtsam::Pose3>(second_camera_pose_key);

    // get frame ground truth - we only store thre previous frame result
    CHECK(frame_id_result_map.find(second_frame - 1) != frame_id_result_map.end());
    ObjectMotionResults& first_frame_object_motion_results = frame_id_result_map.at(second_frame - 1);

    // there are a bunch of potential options in each frame. We need to get all the points associated with the motion
    // at the first and second frames and see which lie within the bounding box
    gtsam::Point3 first_frame_point_world =
        initial_values.at<gtsam::Point3>(landmark_tenrary_factor->PreviousPointKey());
    gtsam::Point3 first_frame_point_camera = first_frame_camera_pose.transformTo(first_frame_point_world);

    // this handles some edge case where dynamic points are not atttached to the motion factor
    // I only see this on kitti 0018 and 0020
    // a similar edge case (or rather the same one) is handled in the graphDeconstruction function
    // when handling the dynamic Point3DFactors
    if (!graph_deconstruction.tracklets.hasFeature(landmark_tenrary_factor->PreviousPointKey()) ||
        !graph_deconstruction.tracklets.hasFeature(landmark_tenrary_factor->CurrentPointKey()))
    {
      continue;
    }

    // just a quick sanity check for the TrackletMap -> we will also update these variables with object id's
    DynamicFeature::shared_ptr first_frame_feature_camera = std::dynamic_pointer_cast<DynamicFeature>(
        graph_deconstruction.tracklets.atFeature(landmark_tenrary_factor->PreviousPointKey()));
    CHECK(first_frame_feature_camera);
    CHECK_EQ(first_frame_feature_camera->camera_pose_key, first_camera_pose_key);
    CHECK(first_frame_feature_camera->motion_key.is_initialized());  // first feature must always have a motion key
    DynamicFeature::shared_ptr second_frame_feature_camera = std::dynamic_pointer_cast<DynamicFeature>(
        graph_deconstruction.tracklets.atFeature(landmark_tenrary_factor->CurrentPointKey()));
    CHECK(second_frame_feature_camera);
    CHECK_EQ(second_frame_feature_camera->camera_pose_key, second_camera_pose_key);
    CHECK(second_frame_feature_camera->motion_key.is_initialized() ||
          second_frame_feature_camera->is_dynamic_);  // the second feature is allowed to not have a motion key if it is
                                                      // at the end of the sequence
    gtsam::Point3 second_frame_point_world =
        initial_values.at<gtsam::Point3>(landmark_tenrary_factor->CurrentPointKey());
    gtsam::Point3 second_frame_point_camera = second_frame_camera_pose.transformTo(second_frame_point_world);
    dsc::KeypointCV first_frame_kp, second_frame_kp;
    if (!camera.isLandmarkContained(first_frame_point_camera, first_frame_kp) ||
        !camera.isLandmarkContained(second_frame_point_camera, second_frame_kp))
    {
      LOG(ERROR) << "Kp's could not be projected for frame " << second_frame - 1 << " and " << second_frame;
      continue;
    }

    cv::Mat& img = debug_images.at(second_frame - 1);

    dsc::utils::DrawCircleInPlace(img, first_frame_kp.pt, dsc::utils::getObjectColour(1), 1);
    dsc::utils::DrawCircleInPlace(img, second_frame_kp.pt, dsc::utils::getObjectColour(1), 1);

    ObjectMotionResult::shared_ptr associated_result = nullptr;
    // LOG(INFO) << first_frame_object_motion_results.size();
    int found_associations = 0;
    for (ObjectMotionResult::shared_ptr o : first_frame_object_motion_results)
    {
      // const auto& bb = o->first_frame_bounding_box;
      // LOG(INFO) << bb.b1 << " " << bb.b2 << " " << bb.b3 << " " << bb.b4;
      // assume no overlap???
      // this currently means overlap with ANY bounding box which is not what we want - we just want overlap with other
      // points
      if (isKpInBoundingBox(first_frame_kp, o->first_frame_bounding_box) &&
          isKpInBoundingBox(second_frame_kp, o->second_frame_bounding_box))
      {
        found_associations++;
        associated_result = o;
      }
    }

    if (found_associations > 1)
    {
      // LOG(WARNING) << "Multiple associations for object beween " << second_frame - 1 << " and " << second_frame;
      continue;
    }

    // LOG(INFO) << "Found associations " << found_associations;

    // the association is on the motion which will happen
    if (associated_result)
    {
      //   dsc::utils::DrawCircleInPlace(img, first_frame_kp.pt,
      //                                 dsc::Display::getObjectColour(associated_result->object_label), 1);
      // dsc::utils::DrawCircleInPlace(img, second_frame_kp.pt,
      // dsc::Display::getObjectColour(associated_result->object_label), 1);
      associated_result->initial_estimate = initial_values.at<gtsam::Pose3>(landmark_tenrary_factor->MotionKey());
      if (optimized_values)
        associated_result->optimized_estimate =
            optimized_values->at<gtsam::Pose3>(landmark_tenrary_factor->MotionKey());

      first_frame_feature_camera->object_label = associated_result->object_label;
      second_frame_feature_camera->object_label = associated_result->object_label;
      CHECK(associated_result->object_label != 0);
      CHECK(first_frame_feature_camera->isDynamic());
      CHECK(second_frame_feature_camera->isDynamic());
      associated_result->has_estimated = true;
    }
    else
    {
      first_frame_feature_camera->object_label = boost::none;
      second_frame_feature_camera->object_label = boost::none;
      // first_frame_feature_camera->motion_key = boost::none;
      // second_frame_feature_camera->motion_key = boost::none;
    }
  }

  for (auto& it : object_result_map)
  {
    ObjectMotionResults& object_motions = it.second;
    object_motions.erase(std::remove_if(object_motions.begin(), object_motions.end(),
                                        [](ObjectMotionResult::shared_ptr x) { return !x->has_estimated; }),
                         object_motions.end());
  }

  // for (const auto& it : debug_images)
  // {
  //   cv::imshow("Debug Image", it.second);
  //   cv::waitKey(1000);
  // }

  // print some stats
  for (const auto& it : object_result_map)
  {
    LOG(INFO) << "Object label " << it.first << " with " << it.second.size() << " motions";
  }

  return object_result_map;
}



void writeOutObjectPoints(int gt_object_id, const TrackletMap& tracklets, const ObjectManager& object_manager_object_centric,
                          object_centric::ObjectIDManager& object_id_manager, const gtsam::Values& dyna_values,
                          const gtsam::Values& world_centric_values, const WorldObjectCentricLikeMapping& mapping,
                          const std::string& path = getOutputPath());
