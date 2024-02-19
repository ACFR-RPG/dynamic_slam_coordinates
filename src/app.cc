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

#include "app.h"

#include <glog/logging.h>
#include <gflags/gflags.h>


#include "Common.h"
#include "ObjectCentric.h"
#include "Optimizer.h"
#include "DataProvider.h"
#include "UtilsOpenCV.h"
#include "UtilsGtsam.h"
#include "GtsamIO.h"
#include "Metrics.h"
#include <opencv2/opencv.hpp>

#include <map>

#include <random>
#include <cstdlib>
#include <signal.h>
#include <iostream>
#include <vector>

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/inference/Key.h>
#include <gtsam/inference/inferenceExceptions.h>

#include <filesystem>

DECLARE_bool(init_motion_identity);
DECLARE_bool(use_object_odometry_factor);
DECLARE_bool(use_object_centric_motion_factor);
DECLARE_double(object_centric_point_factor_sigma);
DECLARE_double(object_centric_motion_factor_sigma);
DECLARE_double(object_odometry_sigma);

DEFINE_int32(input_type, 0, "input type of dataset; 0 = kitti, 2 = sim");
DEFINE_string(kitti_dataset, "", "number of kitti dataset. Must be set if input_type is 0 - this is the fodler name under the path_to_kitti");
DEFINE_string(path_to_kitti, "/root/data/kitti", "Path to KITTI dataset");

DEFINE_bool(run_world_centric_opt, true, "Run the optimization on the input (vdo) graph file");
DEFINE_bool(run_object_centric_opt, true, "Construct and run the optimization on the Object Centric system");


DEFINE_string(output_path, "/root/results/", "Output path to write logs too");
DEFINE_string(graph_file_folder_path, "/root/graphfiles/", "Folder where input graph files will be stored."
    " The expected file structure of an input graph file is then  FLAGS_graph_file_folder_path + / + "
    " dynamic_slam_graph_before_opt_kitti_<FLAGS_kitti_dataset>_motion.g2o");

DEFINE_bool(run_dynamic_experiments, true,
            "If true, runs the dynamic experiments with hardcoded file paths. Otherwise, takes in graph file input "
            "from the cmd line. Still requires path to datasets, rather than a path to a file");

/**
 * @brief Get the output path (as set by gflags) as a valid path and checks if the folder exists
 * 
 * @return std::string 
 */
std::string getOutputPath() {
    namespace fs = std::filesystem;

    const auto output_path = fs::path(FLAGS_output_path);
    if(!fs::exists(output_path)) {
        throw std::runtime_error("Output path does not exist: " + std::string(output_path));
    }
    return output_path;
}

std::string constructOutputPathFile(const std::string file_name) {
    namespace fs = std::filesystem;
    fs::path out_path = getOutputPath();
    return out_path / file_name;
}


/**
 * @brief Constructs the path to a kitti sequence using the gflag variable FLAGS_path_to_kitti as the folder root.
 * The requested kitti_dataset name (e.g. 0000, 0004....) is then appended to the folder and the full path is checked for
 * existance. 
 * 
 * The gflag FLAGS_kitti_dataset is used as the default kitti_dataset argument
 * 
 * @param kitti_dataset 
 * @return std::string 
 */
std::string getKittiSequencePath(const std::string& kitti_dataset = FLAGS_kitti_dataset) {
    namespace fs = std::filesystem;

    const auto dataset_path = fs::path(FLAGS_path_to_kitti);
    const auto sequence_path = dataset_path / kitti_dataset;

     if(!fs::exists(sequence_path)) {
        throw std::runtime_error("Path to kitti sequence does not exist: " + std::string(sequence_path));
    }
    return sequence_path;
}

/**
 * @brief Constructs the path to an input graph file using the gflag variable FLAGS_graph_file_folder_path as the folder root.
 * The requested kitti_dataset name (e.g. 0000, 0004....) is then used to make the expected name if the input file which is in the form
 * "dynamic_slam_graph_before_opt_kitti_<kitti_dataset>"_motion.g2o"
 * 
 * The gflag FLAGS_kitti_dataset is used as the default kitti_dataset argument
 * 
 * @param kitti_dataset Should be like the folder name when constructing the sequence path (e.g. 0000, 0004...)
 * @return std::string 
 */
std::string getGraphFilePath(const std::string& kitti_dataset = FLAGS_kitti_dataset) {
    namespace fs = std::filesystem;

    const auto graphfiles_path = fs::path(FLAGS_graph_file_folder_path);

    const std::string graph_file_name = "dynamic_slam_graph_before_opt_kitti_" + kitti_dataset + + "_motion.g2o";
    const auto file_path = graphfiles_path / graph_file_name;
    if(!fs::exists(file_path)) {
        throw std::runtime_error("Path to graph file does not exist: " + std::string(file_path));
    }
    return file_path;
}

/**
 * @brief Constructs the path to an input graph file where the dataset is represented as an int not a string
 * e.g. to load the graph file path for sequence "0000" -> use 0, "0004", use 4 etc
 * 
 * @param kitti_dataset 
 * @return std::string 
 */
std::string getGraphFilePath(int kitti_dataset) {
    namespace fs = std::filesystem;

    std::stringstream ss;
    ss << std::setfill('0') << std::setw(4) << kitti_dataset;
    return getGraphFilePath(ss.str());    
}

void writeOutObjectPoints(int gt_object_id, const TrackletMap& tracklets, const ObjectManager& object_manager_object_centric,
                          object_centric::ObjectIDManager& object_id_manager, const gtsam::Values& object_centric_values,
                          const gtsam::Values& world_centric_values, const WorldObjectCentricLikeMapping& mapping, const std::string& path)
{
  LOG(INFO) << "Starting write out for object points - " << gt_object_id;

  const std::string objfile = path + "/object_" + std::to_string(gt_object_id) + "_points.txt";
  std::ofstream obj_stream;
  obj_stream.open(objfile);

  // this gives me world-centric camera key
  FrameToKeyMap cam_pose_to_feature_key = tracklets.reorderToPerFrame();

  // start at 1 to match VDO formatting
  size_t current_frame_id = 0;

  for (const auto& it : cam_pose_to_feature_key)
  {
    current_frame_id++;
    const gtsam::KeyVector& feature_keys = it.second;


    const auto& object_centric_object_manager_it = object_manager_object_centric.find(current_frame_id);
    if (object_centric_object_manager_it == object_manager_object_centric.end())
    {
      continue;
    }
    const ObjectObservations& object_centric_object_obs = object_centric_object_manager_it->second;

    std::vector<int> object_centric_labeled_objects = object_centric_object_obs.ObservedObjects();

    int object_centric_object_label = -1;

    for (const int object_centric_label : object_centric_labeled_objects)
    {
      if (object_centric_label == 0)
      {  // background
        continue;
      }


      try
      {
        if (object_id_manager.getGtLabel(object_centric_label) == gt_object_id)
        {
          object_centric_object_label = object_centric_label;
          break;
        }
      }
      catch (const std::out_of_range& e)
      {
        LOG(FATAL) << "Out of bounds when accessing get gt label=" << gt_object_id << ", dyna label=" << object_centric_label;
      }
    }

    if (object_centric_object_label == -1)
    {
      continue;
    }

    int count = 0;
    for (const auto& feature_key : feature_keys)
    {
      const gtsam::Key& world_centric_key = feature_key;
      if (!tracklets.hasFeature(feature_key))
      {
        continue;
      }

      Feature::shared_ptr feature = tracklets.atFeature(feature_key);
      if (feature->isDynamic())
      {
        auto dynamic_feature = std::dynamic_pointer_cast<DynamicFeature>(feature);
        CHECK(dynamic_feature);

        if (dynamic_feature->object_label.get() == gt_object_id)
        {
          try
          {
            gtsam::Key object_key_object_centric = ObjectPoseKey(object_centric_object_label, current_frame_id);
            gtsam::Pose3 object_pose = object_centric_values.at<gtsam::Pose3>(object_key_object_centric);
            const gtsam::Key object_centric_key = mapping.getObjectCentricKey(world_centric_key);

            const gtsam::Point3& object_centric_point = object_centric_values.at<gtsam::Point3>(object_centric_key);
            const gtsam::Point3 object_centric_point_world = object_pose * object_centric_point;

            const gtsam::Point3& point_world = world_centric_values.at<gtsam::Point3>(world_centric_key);
            count++;

            obj_stream << current_frame_id << " " << point_world.x() << " " << point_world.y() << " " << point_world.z()
                       << " " << object_centric_point_world.x() << " " << object_centric_point_world.y() << " " << object_centric_point_world.z()
                       << '\n';
          }
          catch (const std::out_of_range&)
          {
          }
          catch (const gtsam::ValuesKeyDoesNotExist& e)
          {
            LOG(FATAL) << "They key " << ObjectCentricKeyFormatter(e.key()) << " does not exist in the values";
          }
        }
      }
    }
    LOG(INFO) << "Written out " << count << " points for object " << gt_object_id << " - frame " << current_frame_id;
  }
}

std::string save_camera_pose_result(const CameraPoseResultMap& cam_pose_result_map,
                                    const std::string initial_file = "/initial_xyz_poses.txt",
                                    const std::string opt_file = "/optimized_xyz_poses.txt",
                                    const std::string& path = getOutputPath())
{
  const std::string optimized_path = path + opt_file;
  const std::string initial_path = path + initial_file;

  std::ofstream optimized_steam, initial_stream;
  optimized_steam.open(optimized_path);
  initial_stream.open(initial_path);
  LOG(INFO) << "saving xyz pose values to path - " << path;

  dsc::ErrorPairVector initial_absolute_camera_error, initial_relative_camera_error;
  dsc::ErrorPairVector opt_absolute_camera_error, opt_relative_camera_error;

  for (const auto& it : cam_pose_result_map)
  {
    const CameraPoseResult& result = it.second;
    const auto& opt_pose = result.optimized_estimate;
    const auto& initial_pose = result.initial_estimate;
    const auto& gt_pose = result.ground_truth;
    const int frame_id = it.first;

    optimized_steam << "POSE3 " << opt_pose.x() << " " << opt_pose.y() << " " << opt_pose.z() << " " << gt_pose.x()
                    << " " << gt_pose.y() << " " << gt_pose.z() << "\n";
    initial_stream << "POSE3 " << initial_pose.x() << " " << initial_pose.y() << " " << initial_pose.z() << " "
                   << gt_pose.x() << " " << gt_pose.y() << " " << gt_pose.z() << "\n";

    double t_error_before_opt, r_error_before_opt, t_error_after_opt, r_error_after_opt;
    double rel_t_error_before_opt, rel_r_error_before_opt, rel_t_error_after_opt, rel_r_error_after_opt;

    // frames here start at 1
    if (frame_id > 1)
    {
      const CameraPoseResult& prev_result = cam_pose_result_map.at(frame_id - 1);
      const auto& opt_prev_pose = prev_result.optimized_estimate;
      const auto& initial_prev_pose = prev_result.initial_estimate;
      const auto& gt_prev_pose = prev_result.ground_truth;

      // initial relative
      dsc::calculateRelativePoseError(initial_prev_pose, initial_pose, gt_prev_pose, gt_pose, rel_t_error_before_opt,
                                      rel_r_error_before_opt);
      initial_relative_camera_error.push_back(dsc::ErrorPair(rel_t_error_before_opt, rel_r_error_before_opt));

      // optimized relative
      dsc::calculateRelativePoseError(opt_prev_pose, opt_pose, gt_prev_pose, gt_pose, rel_t_error_after_opt,
                                      rel_r_error_after_opt);
      opt_relative_camera_error.push_back(dsc::ErrorPair(rel_t_error_after_opt, rel_r_error_after_opt));
    }

    // initial absolute
    dsc::calculatePoseError(initial_pose, gt_pose, t_error_before_opt, r_error_before_opt);
    initial_absolute_camera_error.push_back(dsc::ErrorPair(t_error_before_opt, r_error_before_opt));

    // optimized absolute
    dsc::calculatePoseError(opt_pose, gt_pose, t_error_after_opt, r_error_after_opt);
    opt_absolute_camera_error.push_back(dsc::ErrorPair(t_error_after_opt, r_error_after_opt));
  }

  auto initial_initial_absolute_camera_error_pair = initial_absolute_camera_error.average();
  auto initial_relative_camera_error_pair = initial_relative_camera_error.average();

  auto opt_absolute_camera_error_pair = opt_absolute_camera_error.average();
  auto opt_relative_camera_error_pair = opt_relative_camera_error.average();

  std::stringstream ss;
  ss << "Initial absolute camera error: t - " << initial_initial_absolute_camera_error_pair.translation << " r - "
     << initial_initial_absolute_camera_error_pair.rot << "\n";
  ss << "Initial relative camera error: t - " << initial_relative_camera_error_pair.translation << " r - "
     << initial_relative_camera_error_pair.rot << "\n";
  ss << "Opt absolute camera error: t - " << opt_absolute_camera_error_pair.translation << " r - "
     << opt_absolute_camera_error_pair.rot << "\n";
  ss << "Opt relative camera error: t - " << opt_relative_camera_error_pair.translation << " r - "
     << opt_relative_camera_error_pair.rot << "\n";

  return ss.str();
}

CameraPoseResultMap constructCameraPoseResultMap(dsc::DataProvider& dataprovider,
                                                 const GraphDeconstruction& graph_deconstruction,
                                                 const gtsam::NonlinearFactorGraph& graph,
                                                 const gtsam::Values& initial_values,
                                                 const gtsam::Values& optimized_values)
{
  dataprovider.reset();
  gtsam::KeyVector cam_key_vector(graph_deconstruction.camera_pose_keys.size());
  std::copy(graph_deconstruction.camera_pose_keys.begin(), graph_deconstruction.camera_pose_keys.end(),
            cam_key_vector.begin());

  CHECK_EQ(cam_key_vector.size(), dataprovider.size());

  dsc::InputPacket input;
  dsc::GroundTruthInputPacket gt;

  CameraPoseResultMap cam_pose_map;

  for (size_t i = 0; i < cam_key_vector.size(); i++)
  {
    const gtsam::Key& cam_key = cam_key_vector.at(i);
    CHECK(dataprovider.next(input, gt));

    CameraPoseResult result;
    result.ground_truth = gt.X_wc;
    result.initial_estimate = initial_values.at<gtsam::Pose3>(cam_key);
    result.optimized_estimate = optimized_values.at<gtsam::Pose3>(cam_key);

    cam_pose_map[gt.frame_id] = result;
  }
  return cam_pose_map;
}

CameraPoseResultMap constructCameraPoseResultMap(const gtsam::Values& gt_values,
                                                 const GraphDeconstruction& graph_deconstruction,
                                                 const gtsam::NonlinearFactorGraph& graph,
                                                 const gtsam::Values& initial_values,
                                                 const gtsam::Values& optimized_values)
{
  gtsam::KeyVector cam_key_vector(graph_deconstruction.camera_pose_keys.size());
  std::copy(graph_deconstruction.camera_pose_keys.begin(), graph_deconstruction.camera_pose_keys.end(),
            cam_key_vector.begin());

  CameraPoseResultMap cam_pose_map;

  for (size_t i = 0; i < cam_key_vector.size(); i++)
  {
    const gtsam::Key& cam_key = cam_key_vector.at(i);

    CameraPoseResult result;
    result.ground_truth = gt_values.at<gtsam::Pose3>(cam_key);
    result.initial_estimate = initial_values.at<gtsam::Pose3>(cam_key);
    result.optimized_estimate = optimized_values.at<gtsam::Pose3>(cam_key);

    // we index frames from 1
    cam_pose_map[i + 1] = result;
  }
  return cam_pose_map;
}

ObjectLabelResultMap constructObjectMotionResultMap(const gtsam::Values& gt_values,
                                                    const GraphDeconstruction& graph_deconstruction,
                                                    const gtsam::NonlinearFactorGraph& graph,
                                                    const gtsam::Values& initial_values,
                                                    const gtsam::Values& optimized_values)
{
  // here we will use the object id as part of the landmark ternary factor as probvided by the VDOTestingGraph file
  ObjectLabelResultMap object_result_map;

  for (size_t i = 0; i < graph.size(); i++)
  {
    const gtsam::NonlinearFactor::shared_ptr& factor = graph.at(i);
    // const gtsam::KeyVector& keys = factor->keys();

    const auto& landmark_tenrary_factor = boost::dynamic_pointer_cast<dsc::LandmarkMotionTernaryFactor>(factor);
    if (landmark_tenrary_factor)
    {
      // Directly from extra information from the graph file at the end of the factor
      int object_id = landmark_tenrary_factor->object_label_;

      if (object_result_map.find(object_id) == object_result_map.end())
      {
        // is a new object
        // LOG(INFO) << "New object label " << object_id;
        object_result_map[object_id] = ObjectMotionResults();
      }

      ObjectMotionResult::shared_ptr this_object_motion = std::make_shared<ObjectMotionResult>();
      this_object_motion->initial_estimate = initial_values.at<gtsam::Pose3>(landmark_tenrary_factor->MotionKey());
      this_object_motion->optimized_estimate = optimized_values.at<gtsam::Pose3>(landmark_tenrary_factor->MotionKey());
      this_object_motion->ground_truth_W = gt_values.at<gtsam::Pose3>(landmark_tenrary_factor->MotionKey());
      this_object_motion->has_estimated = true;
      object_result_map[object_id].push_back(this_object_motion);
    }
  }

  return object_result_map;
}

std::string save_object_motion_result(const ObjectLabelResultMap& object_motion_result_map, bool use_composed,
                                      const ReferenceFrame motion_frame,
                                      const std::string& object_pose_initial_file = "/object_pose.txt",
                                      const std::string& object_pose_refined_file = "/object_pose_refined.txt",
                                      const std::string& object_motion_file = "/object_motion_error.txt",
                                      const std::string& path = getOutputPath())
{
  if (motion_frame == ReferenceFrame::CAMERA_CURRENT || motion_frame == ReferenceFrame::OBJECT_CURRENT)
  {
    throw std::runtime_error(
        "Cannot evaluation motion in camera current reference frame. Must be world, camera previous or object "
        "previous");
  }

  const std::string initial_pose_path = path + object_pose_initial_file;
  const std::string refined_pose_path = path + object_pose_refined_file;
  std::ofstream initial_pose_file, refined_pose_file;
  initial_pose_file.open(initial_pose_path);
  refined_pose_file.open(refined_pose_path);

  const std::string error_path = path + object_motion_file;
  std::ofstream error_file;
  error_file.open(error_path);

  std::stringstream result_str;
  result_str << "Using composed - " << use_composed << "\n";

  dsc::ErrorPairVector total_object_error, total_object_pose_rel_error;
  for (const auto& it : object_motion_result_map)
  {
    int object_id = it.first;
    if (it.second.size() < 5)
    {
      continue;
    }

    dsc::ErrorPairVector initial_error, opt_error, initial_pose_rel_error, initial_pose_abs_error, opt_pose_rel_error,
        opt_pose_abs_error;
    gtsam::Pose3 composed_pose_refined, composed_pose_initial;
    bool is_first = true;
    int object_poses_found = 0;
    for (ObjectMotionResult::shared_ptr result : it.second)
    {
      // motion errors
      double t_error_before_opt, r_error_before_opt, t_error_after_opt, r_error_after_opt;

      // pose errors
      double pose_t_error_before_opt, pose_r_error_before_opt, pose_t_error_after_opt, pose_r_error_after_opt;
      double pose_rel_t_error_before_opt, pose_rel_r_error_before_opt, pose_rel_t_error_after_opt,
          pose_rel_r_error_after_opt;

      CHECK(result->has_estimated);

      // frame agnostic
      const gtsam::Pose3 opt_estimate_motion = result->optimized_estimate;
      const gtsam::Pose3 initial_estimate_motion = result->initial_estimate;

      gtsam::Pose3 opt_estimate_motion_w;
      gtsam::Pose3 initial_estimate_motion_w;

      if (motion_frame == ReferenceFrame::WORLD)
      {
        opt_estimate_motion_w = opt_estimate_motion;
        initial_estimate_motion_w = initial_estimate_motion;
      }
      else if (motion_frame == ReferenceFrame::CAMERA_PREVIOUS)
      {
        const gtsam::Pose3 cam_pose_previous = result->first_frame_camera_pose_gt;
        opt_estimate_motion_w = cam_pose_previous * opt_estimate_motion * cam_pose_previous.inverse();
        initial_estimate_motion_w = cam_pose_previous * initial_estimate_motion * cam_pose_previous.inverse();
      }
      else if (motion_frame == ReferenceFrame::OBJECT_PREVIOUS)
      {
        const gtsam::Pose3 object_pose_previous = result->object_pose_first_world_gt;
        opt_estimate_motion_w = object_pose_previous * opt_estimate_motion * object_pose_previous.inverse();
        initial_estimate_motion_w = object_pose_previous * initial_estimate_motion * object_pose_previous.inverse();
      }

      // LOG(INFO) << result->initial_estimate << " " << opt_estimate_motion << " " << result->ground_truth;
      dsc::calculatePoseError(initial_estimate_motion_w, result->ground_truth_W, t_error_before_opt,
                              r_error_before_opt);

      dsc::calculatePoseError(opt_estimate_motion_w, result->ground_truth_W, t_error_after_opt, r_error_after_opt);

      initial_error.push_back(dsc::ErrorPair(t_error_before_opt, r_error_before_opt));
      opt_error.push_back(dsc::ErrorPair(t_error_after_opt, r_error_after_opt));

      // save in form frame id object id t_error_before r_error_before t_error_after r_error_after
      error_file << "MOTION_ERRORS " << result->frame_id << " " << result->object_label;
      error_file << " " << t_error_before_opt << " " << r_error_before_opt;
      error_file << " " << t_error_after_opt << " " << r_error_after_opt << "\n";

      gtsam::Pose3 object_pose_refined, object_pose_initial;
      gtsam::Pose3 gt_pose = result->object_pose_second_world_gt;

      if (use_composed)
      {
        if (is_first)
        {
          composed_pose_refined = gt_pose;
          composed_pose_initial = gt_pose;
          is_first = false;
        }
        else
        {
          // in camera (convention)
          composed_pose_refined = opt_estimate_motion_w * composed_pose_refined;
          composed_pose_initial = initial_estimate_motion_w * composed_pose_initial;
        }

        object_pose_refined = composed_pose_refined;
        object_pose_initial = composed_pose_initial;

        // set result pose estimates so we can use them for relative error
        result->initial_estimate_L = object_pose_initial;
        result->optimized_estimate_L = object_pose_refined;
      }
      else
      {
        object_pose_refined = result->optimized_estimate_L.get();
        object_pose_initial = result->initial_estimate_L.get();
      }

      // // maybe should be object_pose_second_world_gt
      initial_pose_file << "OBJECT_POSE3 " << result->frame_id << " " << result->object_label;
      initial_pose_file << " " << object_pose_initial.x() << " " << object_pose_initial.y() << " "
                        << object_pose_initial.z() << " " << gt_pose.x() << " " << gt_pose.y() << " " << gt_pose.z()
                        << "\n";

      refined_pose_file << "OBJECT_POSE3 " << result->frame_id << " " << result->object_label;
      refined_pose_file << " " << object_pose_refined.x() << " " << object_pose_refined.y() << " "
                        << object_pose_refined.z() << " " << gt_pose.x() << " " << gt_pose.y() << " " << gt_pose.z()
                        << "\n";

      // find previous pose
      auto prev_it =
          std::find_if(it.second.begin(), it.second.end(), FindFrameIdObjectMotionResult(result->frame_id - 1));
      if (prev_it != it.second.end())
      {
        ObjectMotionResult::shared_ptr prev_result = *prev_it;

        gtsam::Pose3 prev_object_pose_gt = prev_result->object_pose_second_world_gt;
        gtsam::Pose3 prev_object_pose_initial = prev_result->initial_estimate_L.get();
        gtsam::Pose3 prev_object_pose_refined = prev_result->optimized_estimate_L.get();

        // initial pose relative error
        dsc::calculateRelativePoseError(prev_object_pose_initial, object_pose_initial, prev_object_pose_gt, gt_pose,
                                        pose_rel_t_error_before_opt, pose_rel_r_error_before_opt);

        initial_pose_rel_error.push_back(dsc::ErrorPair(pose_rel_t_error_before_opt, pose_rel_r_error_before_opt));

        // optimized pose relative error
        dsc::calculateRelativePoseError(prev_object_pose_refined, object_pose_refined, prev_object_pose_gt, gt_pose,
                                        pose_rel_t_error_after_opt, pose_rel_r_error_after_opt);
        opt_pose_rel_error.push_back(dsc::ErrorPair(pose_rel_t_error_after_opt, pose_rel_r_error_after_opt));

        // initial pose abs error
        dsc::calculatePoseError(object_pose_initial, gt_pose, pose_t_error_before_opt, pose_r_error_before_opt);
        initial_pose_abs_error.push_back(dsc::ErrorPair(pose_t_error_before_opt, pose_r_error_before_opt));

        // optimized pose abs error
        dsc::calculatePoseError(object_pose_refined, gt_pose, pose_t_error_after_opt, pose_r_error_after_opt);
        opt_pose_abs_error.push_back(dsc::ErrorPair(pose_t_error_after_opt, pose_r_error_after_opt));
      }
    }

    if (it.second.size() > 5)
    {
      // motion
      dsc::ErrorPair initial_error_pair = initial_error.average();
      dsc::ErrorPair opt_error_pair = opt_error.average();

      // pose error
      dsc::ErrorPair initial_pose_rel_error_pair = initial_pose_rel_error.average();
      dsc::ErrorPair opt_pose_rel_error_pair = opt_pose_rel_error.average();
      dsc::ErrorPair initial_pose_abs_error_pair = initial_pose_abs_error.average();
      dsc::ErrorPair opt_pose_abs_error_pair = opt_pose_abs_error.average();

      // total_object_error.translation += opt_error.translation;
      // total_object_error.rot += opt_error.rot;
      total_object_error.push_back(dsc::ErrorPair(opt_error_pair.translation, opt_error_pair.rot));
      total_object_pose_rel_error.push_back(
          dsc::ErrorPair(opt_pose_rel_error_pair.translation, opt_pose_rel_error_pair.rot));

      result_str << "Object motion errors " << object_id << " before: t - " << initial_error_pair.translation << " r - "
                 << initial_error_pair.rot << " after t - " << opt_error_pair.translation << " r - "
                 << opt_error_pair.rot << "\n";
      result_str << "Object pose error relative " << object_id << " before: t - "
                 << initial_pose_rel_error_pair.translation << " r - " << initial_pose_rel_error_pair.rot
                 << " after: t - " << opt_pose_rel_error_pair.translation << " r - " << opt_pose_rel_error_pair.rot
                 << "\n";
      result_str << "Object pose error absolute " << object_id << " before: t - "
                 << initial_pose_abs_error_pair.translation << " r - " << initial_pose_abs_error_pair.rot
                 << " after: t - " << opt_pose_abs_error_pair.translation << " r - " << opt_pose_abs_error_pair.rot
                 << "\n";
    }
  }

  // total_object_error.translation /= n_object_used;
  // total_object_error.rot /= n_object_used;
  dsc::ErrorPair total_motion_error_pair = total_object_error.average();
  dsc::ErrorPair total_pose_rel_error_pair = total_object_pose_rel_error.average();

  result_str << "Total object motion errors t - " << total_motion_error_pair.translation << " r - "
             << total_motion_error_pair.rot << "\n";
  result_str << "Total rel object pose errors t - " << total_pose_rel_error_pair.translation << " r - "
             << total_pose_rel_error_pair.rot << "\n";

  initial_pose_file.close();
  error_file.close();
  refined_pose_file.close();

  return result_str.str();
}

void printErrors(dsc::DataProvider& dataprovider, const GraphDeconstruction& graph_deconstruction,
                 const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& initial_values,
                 const gtsam::Values& optimized_values)
{
  dataprovider.reset();
  gtsam::KeyVector cam_key_vector(graph_deconstruction.camera_pose_keys.size());
  std::copy(graph_deconstruction.camera_pose_keys.begin(), graph_deconstruction.camera_pose_keys.end(),
            cam_key_vector.begin());

  CHECK_EQ(cam_key_vector.size(), dataprovider.size());

  dsc::InputPacket input;
  dsc::GroundTruthInputPacket gt;
  dsc::GroundTruthInputPacket previous_gt;

  dsc::ErrorPair initial_absolute_camera_error, initial_relative_camera_error;
  dsc::ErrorPair opt_absolute_camera_error, opt_relative_camera_error;
  for (size_t i = 0; i < cam_key_vector.size(); i++)
  {
    const gtsam::Key& cam_key = cam_key_vector.at(i);
    CHECK(dataprovider.next(input, gt));

    double t_error_before_opt, r_error_before_opt, t_error_after_opt, r_error_after_opt;
    double rel_t_error_before_opt, rel_r_error_before_opt, rel_t_error_after_opt, rel_r_error_after_opt;

    if (i > 0)
    {
      // we can calculate relative errors
      const gtsam::Key& previous_cam_key = cam_key_vector.at(i - 1);
      const gtsam::Pose3& previous_initial_camera_pose = initial_values.at<gtsam::Pose3>(previous_cam_key);
      const gtsam::Pose3& previous_optimized_camera_pose = optimized_values.at<gtsam::Pose3>(previous_cam_key);
      const gtsam::Pose3& previous_gt_camera_pose = previous_gt.X_wc;

      // initial relative
      dsc::calculateRelativePoseError(previous_initial_camera_pose, initial_values.at<gtsam::Pose3>(cam_key),
                                      previous_gt_camera_pose, gt.X_wc, rel_t_error_before_opt, rel_r_error_before_opt);
      initial_relative_camera_error.translation += rel_t_error_before_opt;
      initial_relative_camera_error.rot += rel_r_error_before_opt;

      // optimized relative
      dsc::calculateRelativePoseError(previous_optimized_camera_pose, optimized_values.at<gtsam::Pose3>(cam_key),
                                      previous_gt_camera_pose, gt.X_wc, rel_t_error_after_opt, rel_r_error_after_opt);
      opt_relative_camera_error.translation += rel_t_error_after_opt;
      opt_relative_camera_error.rot += rel_r_error_after_opt;
    }

    // initial absolute
    dsc::calculatePoseError(initial_values.at<gtsam::Pose3>(cam_key), gt.X_wc, t_error_before_opt, r_error_before_opt);
    initial_absolute_camera_error.translation += t_error_before_opt;
    initial_absolute_camera_error.rot += r_error_before_opt;

    // optimized absolute
    dsc::calculatePoseError(optimized_values.at<gtsam::Pose3>(cam_key), gt.X_wc, t_error_after_opt, r_error_after_opt);
    opt_absolute_camera_error.translation += t_error_after_opt;
    opt_absolute_camera_error.rot += r_error_after_opt;

    previous_gt = gt;
  }

  initial_absolute_camera_error.translation /= cam_key_vector.size();
  initial_absolute_camera_error.rot /= cam_key_vector.size();

  initial_relative_camera_error.translation /= (cam_key_vector.size() - 1);
  initial_relative_camera_error.rot /= (cam_key_vector.size() - 1);

  opt_absolute_camera_error.translation /= cam_key_vector.size();
  opt_absolute_camera_error.rot /= cam_key_vector.size();

  opt_relative_camera_error.translation /= (cam_key_vector.size() - 1);
  opt_relative_camera_error.rot /= (cam_key_vector.size() - 1);

  LOG(INFO) << "Initial absolute camera error: t - " << initial_absolute_camera_error.translation << " r - "
            << initial_absolute_camera_error.rot;
  LOG(INFO) << "Initial relative camera error: t - " << initial_relative_camera_error.translation << " r - "
            << initial_relative_camera_error.rot;
  LOG(INFO) << "Opt absolute camera error: t - " << opt_absolute_camera_error.translation << " r - "
            << opt_absolute_camera_error.rot;
  LOG(INFO) << "Opt relative camera error: t - " << opt_relative_camera_error.translation << " r - "
            << opt_relative_camera_error.rot;
}

gtsam::Values resetWorldCentricMotionInit(const GraphDeconstruction& graph_deconstruction, const gtsam::Values& initial)
{
  gtsam::Values new_values = initial;

  for (const auto index : graph_deconstruction.object_motion_keys)
  {
    new_values.update(index, gtsam::Pose3::Identity());
  }
  return new_values;
}

// expect orioginal vdo system so motion is in world
gtsam::Values resetVDOSLAMMotionToFrame(const GraphDeconstruction& graph_deconstruction,
                                        const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& initial,
                                        const ReferenceFrame& to_motion_frame)
{
  gtsam::Values new_values = initial;

  if (to_motion_frame == ReferenceFrame::WORLD)
  {
    return new_values;
  }

  gtsam::KeyVector cam_key_vector(graph_deconstruction.camera_pose_keys.size());
  std::copy(graph_deconstruction.camera_pose_keys.begin(), graph_deconstruction.camera_pose_keys.end(),
            cam_key_vector.begin());
  std::sort(cam_key_vector.begin(), cam_key_vector.end());

  for (const size_t motion_factor_idx : graph_deconstruction.landmark_ternary_factor_idx)
  {
    const auto& landmark_tenrary_factor =
        boost::dynamic_pointer_cast<dsc::LandmarkMotionTernaryFactor>(graph.at(motion_factor_idx));
    CHECK(landmark_tenrary_factor);

    gtsam::Key second_camera_pose_key =
        landmark_tenrary_factor->camera_pose_key_;          // we also save the second camera pose key
    int second_frame = landmark_tenrary_factor->frameId();  // in vdo-slam we save the current id
    int second_frame_idx = second_frame;                    // frame id's start at 0
    CHECK_EQ(cam_key_vector[second_frame_idx], second_camera_pose_key) << " Frame id = " << second_frame_idx;
    int first_frame_idx = second_frame_idx - 1;  // assume frames are consequative?
    CHECK(first_frame_idx >= 0);
    gtsam::Key previous_pose_key = cam_key_vector[first_frame_idx];

    gtsam::Pose3 previous_cam_pose = initial.at<gtsam::Pose3>(previous_pose_key);
    gtsam::Pose3 motion_in_world = initial.at<gtsam::Pose3>(landmark_tenrary_factor->MotionKey());

    if (to_motion_frame == ReferenceFrame::CAMERA_PREVIOUS)
    {
      gtsam::Pose3 motion_in_camera = previous_cam_pose * motion_in_world * previous_cam_pose.inverse();
      new_values.update(landmark_tenrary_factor->MotionKey(), motion_in_camera);
    }
    else
    {
      throw std::runtime_error("Cannot apply conversions of to motion frame");
    }
  }
  return new_values;
}

gtsam::Values runLMOptimizer(const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& initial,
                             std::stringstream& result_str, IterationOutputWriter* output_writer = nullptr,
                             PerInnerIterationOutputWriter* inner_iteration_ow = nullptr)
{
  result_str << "Graph size " << graph.size() << " values size " << initial.size() << "\n";
  LOG(INFO) << "Running OPT";

  std::string system_name;
  if (FLAGS_run_object_centric_opt)
  {
    system_name = getObjectCentricSystemName();
  }
  else if (FLAGS_run_world_centric_opt)
  {
    system_name = "world_centric";
  }

  LOG(INFO) << system_name;


  double error_before = graph.error(initial);
  gtsam::LevenbergMarquardtWithLoggingParams params;
  params.setMaxIterations(100);
  params.verbosity = gtsam::NonlinearOptimizerParams::Verbosity::ERROR;
  params.logFile = constructOutputPathFile(FLAGS_kitti_dataset + "_" + system_name + "_opt_logout.csv");

  if (output_writer)
    params.iterationHook = output_writer->getHook();

  if (inner_iteration_ow)
    params.inner_iteration_hook = inner_iteration_ow->getHook();

  gtsam::LevenbergMarquardtOptimizerWithLogging opt(graph, initial, params);

  try
  {
    auto tic = std::chrono::high_resolution_clock::now();
    gtsam::Values result = opt.optimize();
    auto toc = std::chrono::high_resolution_clock::now();

    auto diff = toc - tic;
    double time_to_run = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count()) / 1e9;

    {
      const std::string timing_file = constructOutputPathFile(FLAGS_kitti_dataset + "_" + system_name + "_time.csv");
      std::ofstream time_os_steam_;
      time_os_steam_.open(timing_file, std::ios::app);

      time_os_steam_ << time_to_run << ",";
    }

    double error_after = graph.error(result);
    result_str << "error before " << error_before << " error after " << error_after << " num iterations "
               << opt.iterations() << "\n";
    return result;
  }
  catch (const gtsam::ValuesKeyDoesNotExist& e)
  {
    const gtsam::Key& var = e.key();
    LOG(FATAL) << "Values key does not exist " << ObjectCentricKeyFormatter(var);
  }
}

gtsam::Values runGNOptimizer(const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& initial,
                             const gtsam::GaussNewtonParams& params, std::stringstream& result_str)
{
  result_str << "Graph size " << graph.size() << " values size " << initial.size() << "\n";
  LOG(INFO) << "Running OPT";

  double error_before = graph.error(initial);
  gtsam::GaussNewtonOptimizer opt(graph, initial, params);
  gtsam::Values result = opt.optimize();
  double error_after = graph.error(result);

  result_str << "error before " << error_before << " error after " << error_after << " num iterations "
             << opt.iterations() << "\n";
  return result;
}




void parseKitti(const std::string& measurement_file, const std::string& result_file = std::string())
{
  std::string path_to_sequence = getKittiSequencePath(FLAGS_kitti_dataset);
  LOG(INFO) << "Path to sequence " << path_to_sequence;



  dsc::KittiSequenceDataProvider data_provider(path_to_sequence);
  data_provider.load();

  std::ifstream myfile;
  myfile.open(measurement_file, std::ios::in);

  gtsam::io::GraphFileParser parser;
  LOG(INFO) << "Loading graph";
  gtsam::io::ValuesFactorGraphPair pair = parser.load(myfile);

  gtsam::Values world_centric_values = pair.first;
  gtsam::NonlinearFactorGraph graph = pair.second;
  myfile.close();

  GraphDeconstruction graph_deconstruction = deconstructGraph(graph);

  gtsam::Values values = world_centric_values;
  if (FLAGS_init_motion_identity)
  {
    values = resetWorldCentricMotionInit(graph_deconstruction, world_centric_values);
  }

  std::stringstream results_str;
  results_str << "Results File:\nDataset - " << FLAGS_kitti_dataset;
  results_str << " - init motion with identity: " << FLAGS_init_motion_identity << "\n";

  gtsam::Values result = values;  // vdo optimization result (or initial values if not run)

  if (FLAGS_run_world_centric_opt)
  {
    LOG(INFO) << "Running world-centric optimization...";
    results_str << "\n World-Centric opt - ";
    // currently hardcoded name
    const std::string output_writer_file = constructOutputPathFile(FLAGS_kitti_dataset + "_world_centric_optimization.csv");
    const std::string inner_step_output_writer_file = constructOutputPathFile(FLAGS_kitti_dataset + "_world_centric_per_step_optimization.csv");
    IterationOutputWriter output_writer(output_writer_file);
    PerInnerIterationOutputWriter inner_step_ow(inner_step_output_writer_file);
    result = runLMOptimizer(graph, values, results_str, &output_writer, &inner_step_ow);
  }

  CameraPoseResultMap camera_pose_result_map =
      constructCameraPoseResultMap(data_provider, graph_deconstruction, graph, values, result);
  // NOTE: this function allso modified the graph_deconstruction object becuase i can be a lazy coder!!!
  ObjectLabelResultMap object_motion_result_map =
      constructObjectMotionResultMap(data_provider, graph_deconstruction, graph, values, result);
  LOG(INFO) << " Constructed object motion result map";

  results_str << "VDO-SAM camera pose results:\n";
  results_str << save_camera_pose_result(
    camera_pose_result_map, 
    "/world_centric_initial_xyz_poses.txt", 
    "/world_centric_optimized_xyz_poses.txt");
  results_str << "\nVDO-SAM object motion results:\n";

  constexpr static bool kUseComposedPose = true;
  results_str << save_object_motion_result(object_motion_result_map, kUseComposedPose, ReferenceFrame::WORLD,
                                           "/world_centric_object_pose_initial.txt", 
                                           "/world_centric_object_pose_refined.txt",
                                           "/world_centric_object_motion_error.txt");
  printErrors(data_provider, graph_deconstruction, graph, values, result);

  if (FLAGS_run_object_centric_opt)
  {
    ObjectManager object_manager_object_centric;
    object_centric::ObjectIDManager object_id_manager;
    WorldObjectCentricLikeMapping vdo_slam_to_object_centric_keys;
    auto object_centric_pair =
        constructDynaSLAMLikeGraph(graph_deconstruction, object_motion_result_map, graph, values,
                                   object_manager_object_centric, object_id_manager, vdo_slam_to_object_centric_keys);
    gtsam::Values object_centric_values = object_centric_pair.first;
    gtsam::NonlinearFactorGraph object_centric_graph = object_centric_pair.second;

    results_str << "\n Object Centric config\n";
    results_str << " - use object odometry factor: " << FLAGS_use_object_odometry_factor << "\n";
    results_str << " - use object centric motion factor: " << FLAGS_use_object_centric_motion_factor << "\n";
    results_str << " - object_centric_point_factor_sigma: " << FLAGS_object_centric_point_factor_sigma << "\n";
    results_str << " - object_centric_motion_factor_sigma: " << FLAGS_object_centric_motion_factor_sigma << "\n";
    results_str << " - use object_odometry_sigma: " << FLAGS_object_odometry_sigma << "\n";
    results_str << "\n Object Centric opt - ";
    const std::string output_writer_file = constructOutputPathFile(FLAGS_kitti_dataset + "_" + getObjectCentricSystemName() +  "_optimization.csv");
    const std::string inner_step_output_writer_file = constructOutputPathFile(FLAGS_kitti_dataset + "_" + getObjectCentricSystemName() +  "_per_step_optimization.csv");
    IterationOutputWriter output_writer(output_writer_file);
    PerInnerIterationOutputWriter inner_step_ow(inner_step_output_writer_file);
    gtsam::Values object_centric_result =
        runLMOptimizer(object_centric_graph, object_centric_values, results_str, &output_writer, &inner_step_ow);

    auto object_centric_result_maps =
        constructDynaLikeResults(camera_pose_result_map, object_motion_result_map, object_manager_object_centric,
                                 object_id_manager, vdo_slam_to_object_centric_keys, object_centric_values, object_centric_result);

    const CameraPoseResultMap object_centric_camera_pose_result_map = object_centric_result_maps.first;
    const ObjectLabelResultMap object_centric_object_motion_result_map = object_centric_result_maps.second;

    results_str << "\nObject Centric camera pose results:\n";
    results_str << save_camera_pose_result(
      object_centric_camera_pose_result_map, 
      "/object_centric_initial_xyz_poses.txt",
      "/object_centric_optimized_xyz_poses.txt");
    results_str << "\nObject Centric object motion results:\n";

    bool kUseComposedPose = false;
    results_str << save_object_motion_result(object_centric_object_motion_result_map, kUseComposedPose, ReferenceFrame::WORLD,
                                             "/" + getObjectCentricSystemName() + "_object_pose_initial.txt", 
                                             "/" + getObjectCentricSystemName() + "_object_pose_refined.txt",
                                             "/" + getObjectCentricSystemName() + "_oject_motion_error.txt");

    kUseComposedPose = true;
    save_object_motion_result(
        object_centric_object_motion_result_map, kUseComposedPose, ReferenceFrame::WORLD, 
        "/" + getObjectCentricSystemName() + "_object_centric_object_pose_initial_composed.txt",
        "/" + getObjectCentricSystemName() + "_object_centric_object_pose_refined_composed.txt", 
        "/" + getObjectCentricSystemName() + "_object_centric_oject_motion_error_composed.txt");

  }


  std::stringstream result_file_name;
  if (result_file.empty())
  {
    const std::string results = FLAGS_kitti_dataset + ".txt";
    result_file_name << constructOutputPathFile(results);
  }
  else
  {
    result_file_name << result_file;
  }
  std::ofstream initial_pose_file(result_file_name.str());
  initial_pose_file << results_str.str();
  initial_pose_file.close();
}


void parseSim(const std::string& measurement_file, const std::string& gt_file = std::string(),
              const std::string& result_directory = ".")
{
  bool has_gt = !gt_file.empty();
  if (has_gt)
  {
    LOG(INFO) << "Using gt file " << gt_file << " for comparison";
  }

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values values, gt_values;

  {
    std::ifstream myfile;
    myfile.open(measurement_file, std::ios::in);

    gtsam::io::GraphFileParser parser;
    gtsam::io::ValuesFactorGraphPair pair = parser.load(myfile);

    values = pair.first;
    graph = pair.second;

    // graph.saveGraph("graph.dot");
    myfile.close();
  }

  // load gt
  if (has_gt)
  {
    std::ifstream myfile;
    myfile.open(gt_file, std::ios::in);

    gtsam::io::GraphFileParser parser;
    gtsam::io::ValuesFactorGraphPair pair = parser.load(myfile);

    gt_values = pair.first;
    myfile.close();

    CHECK_EQ(gt_values.size(), values.size());
  }
}

void parsefromCmd(int argc, char** argv)
{
  const std::string input_graph_file = getGraphFilePath(FLAGS_kitti_dataset);

  if (FLAGS_input_type == 0)
  {
    LOG(INFO) << "Parsing as kitti dataset. kitti_dataset must be set to a valid folder";
    parseKitti(input_graph_file);
    // parse kitti
  }
  else if (FLAGS_input_type == 1)
  {
    LOG(FATAL) << "Omd parsing not implemented";
  }
  else if (FLAGS_input_type == 2)
  {
    LOG(INFO) << "Parsing as simulated dataset";
  }
}

void runExperiments()
{
  // dataset
  std::vector<int> dataset_vector = { 0, 3, 4, 5 };

  for (int dataset : dataset_vector)
  {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(4) << dataset;
    
    FLAGS_kitti_dataset = ss.str();
    const std::string measurement_file = getGraphFilePath(dataset);

    FLAGS_run_world_centric_opt = true;
    FLAGS_run_object_centric_opt = false;

    // world centric
    FLAGS_use_object_centric_motion_factor = true;
    FLAGS_use_object_odometry_factor = false;
    {
      LOG(INFO) << "Running world centric for dataset " << FLAGS_kitti_dataset;
      parseKitti(measurement_file, constructOutputPathFile(FLAGS_kitti_dataset + "_world_centric.txt"));
    }

    FLAGS_run_world_centric_opt = false;
    FLAGS_run_object_centric_opt = true;

    // object_centric
    FLAGS_use_object_centric_motion_factor = true;
    FLAGS_use_object_odometry_factor = false;
    {
      LOG(INFO) << "Running object centric for dataset " << FLAGS_kitti_dataset;
      parseKitti(measurement_file, constructOutputPathFile(FLAGS_kitti_dataset + "_object_centric.txt"));
    }

    // with OOF
    FLAGS_use_object_centric_motion_factor = true;
    FLAGS_use_object_odometry_factor = true;
    {
      LOG(INFO) << "Running object centric with OOF for dataset " << FLAGS_kitti_dataset;
      parseKitti(measurement_file, constructOutputPathFile(FLAGS_kitti_dataset + "_object_centric_with_oof.txt"));
    }

    // only OOF
    FLAGS_use_object_centric_motion_factor = false;
    FLAGS_use_object_odometry_factor = true;
    {
      LOG(INFO) << "Running object centric only OOF for dataset " << FLAGS_kitti_dataset;
      parseKitti(measurement_file, constructOutputPathFile(FLAGS_kitti_dataset + "_object_centric_only_oof.txt"));
    }
  }
}


int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    FLAGS_logtostderr = 1;
    FLAGS_colorlogtostderr = 1;
    FLAGS_log_prefix = 1;

    if (FLAGS_run_dynamic_experiments)
    {
        runExperiments();
    }
    else
    {
        parsefromCmd(argc, argv);
    }
}