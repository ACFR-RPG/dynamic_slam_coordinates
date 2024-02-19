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

#include "ObjectCentric.h"

#include "Point3DFactor.h"
#include "LandmarkMotionTernaryFactor.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/BetweenFactor.h>
#include <glog/logging.h>
#include <gflags/gflags.h>

DEFINE_bool(init_motion_identity, true,
            "If the motion variable in the graph should be initalized with identity (for all systems), or the provided "
            "motion");
DEFINE_bool(use_object_odometry_factor, true,
            "If the object_centric::ObjectOdometryFactor should be used in addition to the landmark motion factor");

DEFINE_bool(use_object_centric_motion_factor, true, "If the object_centric:ObjectCentricMotionFactor should be used");

DEFINE_double(object_centric_point_factor_sigma, 0.5,
              "Sigma used for the covariance of the ObjectCentricPoint3DFactor");
DEFINE_double(object_centric_motion_factor_sigma, 10,
              "Sigma used for the covariance of the ObjectCentricLandmarkMotionFactor");
DEFINE_double(object_odometry_sigma, 0.01, "Sigma used for the covariance of the ObjectOdometryFactor (if used)");

DEFINE_int32(min_object_observations, 5,
             "How many times an object must be seen consequatively before it is added. Minimum is 3");

// 0 - centroid, 1 - composed motion, 2 - gt translation
DEFINE_int32(object_pose_init_method, 0, "Determined how the object pose should be initalized");

DECLARE_bool(run_object_centric_opt);

std::string getObjectCentricSystemName() {
  CHECK(FLAGS_run_object_centric_opt);

  bool is_object_centric =  FLAGS_use_object_centric_motion_factor && !FLAGS_use_object_odometry_factor;
  bool is_with_oof = FLAGS_use_object_centric_motion_factor && FLAGS_use_object_centric_motion_factor;
  bool is_only_oof = !FLAGS_use_object_centric_motion_factor && FLAGS_use_object_odometry_factor;

  CHECK(is_object_centric || is_with_oof || is_only_oof);

  if(is_object_centric) {
    return "object_centric"; 
  }
  else if(is_with_oof) {
    return "object_centric_with_oof";
  }
  else if(is_only_oof) {
    return "object_centric_only_oof";
  }
  else {
    LOG(FATAL) << "Could not determine object_centric system name";
  }
}

namespace object_centric
{

ObjectCentricPoint3DFactor::ObjectCentricPoint3DFactor(gtsam::Key cameraPoseKey, gtsam::Key objectPoseKey,
                                                       gtsam::Key objectPointKey, gtsam::Point3 measured,
                                                       gtsam::SharedNoiseModel model)
  : gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Point3>(model, cameraPoseKey, objectPoseKey,
                                                                        objectPointKey)
  , measured_(measured)
{
}

gtsam::Vector3 ObjectCentricPoint3DFactor::calculateResidual(const gtsam::Point3& m_camera, const gtsam::Pose3& X,
                                                             const gtsam::Pose3& L, const gtsam::Point3& m)
{
  gtsam::Point3 point_world = L.transformFrom(m);
  gtsam::Point3 estimated_camera = X.transformTo(point_world);

  return gtsam::Vector3(estimated_camera - m_camera);
}

gtsam::Vector ObjectCentricPoint3DFactor::evaluateError(const gtsam::Pose3& X, const gtsam::Pose3& L,
                                                        const gtsam::Point3& m, boost::optional<gtsam::Matrix&> J1,
                                                        boost::optional<gtsam::Matrix&> J2,
                                                        boost::optional<gtsam::Matrix&> J3) const
{
  if (J1)
  {
    *J1 = gtsam::numericalDerivative31<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3, gtsam::Point3>(
        std::bind(&ObjectCentricPoint3DFactor::calculateResidual, measured_, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3),
        X, L, m);
    // const gtsam::Point3 m_world = L.transformFrom(m);
    // X.transformTo(m_world, *J1, boost::none);
  }

  if (J2)
  {
    *J2 = gtsam::numericalDerivative32<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3, gtsam::Point3>(
        std::bind(&ObjectCentricPoint3DFactor::calculateResidual, measured_, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3),
        X, L, m);
    // gtsam::Matrix H;
    // L.transformFrom(m, H, boost::none);
    // *J2 = X.inverse().rotation().matrix() * H;
  }

  if (J3)
  {
    *J3 = gtsam::numericalDerivative33<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3, gtsam::Point3>(
        std::bind(&ObjectCentricPoint3DFactor::calculateResidual, measured_, std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3),
        X, L, m);
    // *J3 = (X.inverse() * L).rotation().matrix();
  }

  return calculateResidual(measured_, X, L, m);
}

ObjectOdometryFactor::ObjectOdometryFactor(gtsam::Key motionKey, gtsam::Key previousObjectPoseKey,
                                           gtsam::Key currentObjectPoseKey, gtsam::SharedNoiseModel model)
  : gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(model, motionKey, previousObjectPoseKey,
                                                                       currentObjectPoseKey)
{
}

gtsam::Vector6 ObjectOdometryFactor::calculateResidual(const gtsam::Pose3& H_w, const gtsam::Pose3& L_previous_world,
                                                       const gtsam::Pose3& L_current_world)
{
  gtsam::Pose3 propogated = H_w * L_previous_world;
  gtsam::Pose3 Hx = L_current_world.between(propogated);
  return gtsam::Pose3::Identity().localCoordinates(Hx);
  // return gtsam::traits<gtsam::Pose3>::Local(L_current_world, propogated);
  // return gtsam::Pose3::Logmap(L_current_world.inverse() * H_w * L_previous_world);
}

gtsam::Vector ObjectOdometryFactor::evaluateError(const gtsam::Pose3& H_w, const gtsam::Pose3& L_previous_world,
                                                  const gtsam::Pose3& L_current_world,
                                                  boost::optional<gtsam::Matrix&> J1,
                                                  boost::optional<gtsam::Matrix&> J2,
                                                  boost::optional<gtsam::Matrix&> J3) const
{
  if (J1)
  {
    *J1 = gtsam::numericalDerivative31<gtsam::Vector6, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(
        std::bind(&ObjectOdometryFactor::calculateResidual, std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3),
        H_w, L_previous_world, L_current_world);
  }

  if (J2)
  {
    *J2 = gtsam::numericalDerivative32<gtsam::Vector6, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(
        std::bind(&ObjectOdometryFactor::calculateResidual, std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3),
        H_w, L_previous_world, L_current_world);
  }

  if (J3)
  {
    *J3 = gtsam::numericalDerivative33<gtsam::Vector6, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(
        std::bind(&ObjectOdometryFactor::calculateResidual, std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3),
        H_w, L_previous_world, L_current_world);
  }

  return calculateResidual(H_w, L_previous_world, L_current_world);
}

ObjectCentricLandmarkMotionFactor::ObjectCentricLandmarkMotionFactor(gtsam::Key motionKey,
                                                                     gtsam::Key previousObjectPoseKey,
                                                                     gtsam::Key currentObjectPoseKey,
                                                                     gtsam::Key objectPointKey,
                                                                     gtsam::SharedNoiseModel model)
  : gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Point3>(
        model, motionKey, previousObjectPoseKey, currentObjectPoseKey, objectPointKey)
{
}

gtsam::Vector3 ObjectCentricLandmarkMotionFactor::calculateResidual(const gtsam::Pose3& H_world,
                                                                    const gtsam::Pose3& L_previous_world,
                                                                    const gtsam::Pose3& L_current_world,
                                                                    const gtsam::Point3& point_L)
{
  gtsam::Pose3 L_propogated = H_world * L_previous_world;
  // // // want L_current - L_propogated
  gtsam::Pose3 between = gtsam::traits<gtsam::Pose3>::Between(L_current_world, L_propogated, boost::none, boost::none);

  return gtsam::Vector3(between * point_L);
  // return gtsam::Vector3((L_current_world * point_L) - (H_world * L_previous_world * point_L));
}

gtsam::Vector ObjectCentricLandmarkMotionFactor::evaluateError(
    const gtsam::Pose3& H_w, const gtsam::Pose3& L_previous_world, const gtsam::Pose3& L_current_world,
    const gtsam::Point3& point_L, boost::optional<gtsam::Matrix&> J1, boost::optional<gtsam::Matrix&> J2,
    boost::optional<gtsam::Matrix&> J3, boost::optional<gtsam::Matrix&> J4) const
{
  if (J1)
  {
    *J1 = gtsam::numericalDerivative41<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Point3>(
        std::bind(&ObjectCentricLandmarkMotionFactor::calculateResidual, std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3, std::placeholders::_4),
        H_w, L_previous_world, L_current_world, point_L);

    // gtsam::Point3 v = L_previous_world * point_L;
    // H_w.transformFrom(v, J1, boost::none);
  }

  if (J2)
  {
    *J2 = gtsam::numericalDerivative42<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Point3>(
        std::bind(&ObjectCentricLandmarkMotionFactor::calculateResidual, std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3, std::placeholders::_4),
        H_w, L_previous_world, L_current_world, point_L);
    // // previous world
    // gtsam::Matrix H;
    // L_previous_world.transformFrom(point_L, H, boost::none);

    // // H is 3x6 but we need to premultple by H_w which is SE(3) how to stack these? do we just take the rotation
    // part? *J2 = H_w.matrix().topLeftCorner(3, 3) * H;
  }

  if (J3)
  {
    *J3 = gtsam::numericalDerivative43<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Point3>(
        std::bind(&ObjectCentricLandmarkMotionFactor::calculateResidual, std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3, std::placeholders::_4),
        H_w, L_previous_world, L_current_world, point_L);
    // current world
    // L_current_world.transformFrom(point_L, J3, boost::none);
  }

  if (J4)
  {
    *J4 = gtsam::numericalDerivative44<gtsam::Vector3, gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Point3>(
        std::bind(&ObjectCentricLandmarkMotionFactor::calculateResidual, std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3, std::placeholders::_4),
        H_w, L_previous_world, L_current_world, point_L);

    // gtsam::Pose3 L_propogated = H_w * L_previous_world;
    // gtsam::Pose3 J = gtsam::traits<gtsam::Pose3>::Between(L_current_world, L_propogated, boost::none, boost::none);
    // *J4 = (L_previous_world.inverse() * H_w * L_current_world).rotation().matrix();
  }

  return calculateResidual(H_w, L_previous_world, L_current_world, point_L);
}

int ObjectIDManager::getObjectCentricID(int ground_truth_object_id, bool world_centric_lost_tracking)
{
  if (active_object_ids_.find(ground_truth_object_id) == active_object_ids_.end() || world_centric_lost_tracking)
  {
    // new object or not a new object but tracking lost
    active_object_ids_[ground_truth_object_id] = max_object_id;
    LOG(INFO) << "new label =" << max_object_id << " for gt label=" << ground_truth_object_id
              << " world_centric lost tracking =" << world_centric_lost_tracking;

    max_object_id++;
  }
  int new_object_id = active_object_ids_.at(ground_truth_object_id);
  object_centric_object_ids_to_gt_[new_object_id] = ground_truth_object_id;
  return new_object_id;
}

int ObjectIDManager::getGtLabel(int object_centric_object_label) const
{
  return object_centric_object_ids_to_gt_.at(object_centric_object_label);
}

bool ObjectIDManager::hasDynaLikeID(int ground_truth_object_id) const
{
  return (active_object_ids_.find(ground_truth_object_id) != active_object_ids_.end());
}

}  // namespace object_centric

std::string ObjectCentricKeyFormatter(gtsam::Key key)
{
  const gtsam::LabeledSymbol asLabeledSymbol(key);
  if (asLabeledSymbol.chr() > 0 && asLabeledSymbol.label() > 0)
  {
    auto chr = asLabeledSymbol.chr();
    if (chr == kObjectMotionKey || chr == kObjectPoseKey)
    {
      // convert label back to int
      int object_id = asLabeledSymbol.label() - '0';
      // return std::to_string(chr) + std::to_string(object_id) + "-" + std::to_string(asLabeledSymbol.index());
      char buffer[100];
      snprintf(buffer, 100, "%c%d-%llu", chr, object_id, static_cast<unsigned long long>(asLabeledSymbol.index()));
      return std::string(buffer);
    }
    else
    {
      return (std::string)asLabeledSymbol;
    }
  }

  const gtsam::Symbol asSymbol(key);
  if (asLabeledSymbol.chr() > 0)
  {
    return (std::string)asSymbol;
  }
  else
  {
    return std::to_string(key);
  }
}

using gtsam::symbol_shorthand::M;  // dynamic points
using gtsam::symbol_shorthand::P;  // static points
using gtsam::symbol_shorthand::X;  // camera pose

gtsam::SharedNoiseModel createFromworld_centricNoiseModel(gtsam::SharedNoiseModel model, bool keep_robust = true)
{
  // try if robust
  CHECK_NOTNULL(model);
  gtsam::SharedNoiseModel gaussian_model = nullptr;
  auto robust = boost::dynamic_pointer_cast<gtsam::noiseModel::Robust>(model);
  if (robust)
  {
    gtsam::noiseModel::Robust::shared_ptr r = boost::make_shared<gtsam::noiseModel::Robust>(*robust);

    if (!keep_robust)
    {
      return r->noise();
    }
    else
    {
      gaussian_model = r->noise();
    }
  }
  else
  {
    gaussian_model = model;
  }

  auto gaussian = boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(gaussian_model);
  if (!gaussian)
  {
    throw std::runtime_error("Noise model not robust or gaussian");
  }

  return gtsam::noiseModel::Gaussian::Covariance(gaussian->covariance());
}

gtsam::Pose3 calculateCentroid(const gtsam::KeyVector& object_features, const gtsam::Pose3& camera_pose,
                               const TrackletMap& tracklet_map, const object_centric::ObjectIDManager& object_id_manager,
                               int object_label)
{
  gtsam::Point3 centroid;
  for (const gtsam::Key& key : object_features)
  {
    const Feature::shared_ptr& f = tracklet_map.atFeature(key);
    CHECK(f->isDynamic());

    DynamicFeature::shared_ptr dynamic_feature = std::dynamic_pointer_cast<DynamicFeature>(f);
    CHECK(dynamic_feature);
    CHECK_EQ(*dynamic_feature->object_label, object_id_manager.getGtLabel(object_label));

    // measurement should be in camera frame
    centroid += dynamic_feature->measurement;
  }

  centroid /= object_features.size();

  gtsam::Point3 centroid_world = camera_pose.transformFrom(centroid);
  return gtsam::Pose3(gtsam::Rot3::Identity(), centroid_world);
}

gtsam::Pose3 createPropogatedObjectPose(const gtsam::Pose3& object_motion, const gtsam::Pose3& previous_object_pose,
                                        const gtsam::Pose3& centroid,
                                        boost::optional<gtsam::Pose3> gt_pose = boost::none)
{
  gtsam::Pose3 current_object_pose_propogated = object_motion * previous_object_pose;
  // arrow from previous to current
  gtsam::Point3 translation_vector = centroid.translation() - previous_object_pose.translation();

  // double pitch_angle = atan(translation_vector(2)/-translation_vector(0));
  // double roll_angle = 0;
  // double yaw_angle = atan(translation_vector(2)/translation_vector(1));
  double pitch_angle = atan(translation_vector(2) / translation_vector(0));
  double roll_angle = 0;
  double yaw_angle = atan(translation_vector(0) / translation_vector(1));
  gtsam::Rot3 rotation_change = gtsam::Rot3::Ypr(yaw_angle, pitch_angle, roll_angle);
  // LOG(INFO) << rotation_change;
  gtsam::Rot3 absolute_rotation = previous_object_pose.rotation() * rotation_change;

  const int method = FLAGS_object_pose_init_method;
  if (method == 0)
  {
    return gtsam::Pose3(gtsam::Rot3::Identity(), centroid.translation());
  }
  else if (method == 1)
  {
    return gtsam::Pose3(gtsam::Rot3::Identity(), current_object_pose_propogated.translation());
  }
  else if (method == 2)
  {
    if (gt_pose)
      return gt_pose.get();
    else
    {
      return gtsam::Pose3(gtsam::Rot3::Identity(), centroid.translation());
    }
  }
  else
  {
    CHECK(false) << "Unknown object pose init method";
  }

  // return centroid;
  // return current_object_pose_propogated;
}

void initaliseObjectMotionValue(gtsam::Values& object_centric_values, gtsam::Key key, const gtsam::Pose3& motion)
{
  if (FLAGS_init_motion_identity)
  {
    object_centric_values.insert(key, gtsam::Pose3::Identity());
  }
  else
  {
    object_centric_values.insert(key, motion);
  }
}

void initalisePastObjects(const ObjectLabelResultMap& gt_objects, int current_frame, int first_frame,
                          const ObjectManager& object_manager_object_centric,
                          const object_centric::ObjectIDManager& object_id_manager, const TrackletMap& tracklets,
                          const std::map<int, std::map<int, gtsam::Key>>& world_centric_object_key_map,
                          const int object_label,  // object_centric like
                          const gtsam::Values& world_centric_values, WorldObjectCentricLikeMapping& world_centric_slam_to_dyna_keys,
                          gtsam::Values& object_centric_values, gtsam::NonlinearFactorGraph& object_centric_graph,
                          gtsam::SharedNoiseModel object_motion_noisemodel,
                          gtsam::SharedNoiseModel smoothingfactor_noisemodel,
                          gtsam::SharedNoiseModel landmark_tenraryfactor_noisemodel)
{
  for (int frame = first_frame; frame < current_frame; frame++)
  {
    LOG(INFO) << "back filling object poses/motion - " << frame << "/" << current_frame
              << " object label=" << object_label;
    CHECK(object_manager_object_centric.objectInFrame(frame, object_label));

    const ObjectObservations& object_observations = object_manager_object_centric.at(frame);
    CHECK(object_observations.objectObserved(object_label))
        << "Object " << object_label << " was not seen at " << frame;
    const gtsam::KeyVector object_features = object_observations.at(object_label);

    const gtsam::Key object_centric_object_pose_key = ObjectPoseKey(object_label, frame);
    gtsam::Pose3 camera_pose = object_centric_values.at<gtsam::Pose3>(X(frame));

    const int gt_object_label = object_id_manager.getGtLabel(object_label);
    const ObjectMotionResults& gt_object_motion_results = gt_objects.at(gt_object_label);
    auto it = std::find_if(gt_object_motion_results.begin(), gt_object_motion_results.end(),
                           FindFrameIdObjectMotionResult(frame - 1));
    const ObjectMotionResult::shared_ptr& gt_object_result = *it;
    CHECK(it != gt_object_motion_results.end());

    if (frame == first_frame)
    {
      gtsam::Pose3 initial_object_pose = gt_object_result->object_pose_second_world_gt;
      // reset to identity matrix for rotation
      initial_object_pose = gtsam::Pose3(gtsam::Rot3::Identity(), initial_object_pose.translation());
      // object_centric_values.insert(object_centric_object_pose_key, initial_object_pose);

      // // initalise pose only
      // initial_object_pose = calculateCentroid(object_features, camera_pose, tracklets, object_id_manager,
      // object_label);
      object_centric_values.insert(object_centric_object_pose_key, initial_object_pose);

      object_centric_graph.addPrior<gtsam::Pose3>(
          object_centric_object_pose_key, initial_object_pose,
          gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 0.2, 0.2, 0.2, 0.4, 0.4, 0.4).finished()));
    }
    else
    {
      const gtsam::Key object_centric_previous_object_pose_key = ObjectPoseKey(object_label, frame - 1);
      const ObjectObservations& previous_object_observations = object_manager_object_centric.at(frame - 1);
      gtsam::Key object_centric_previous_motion_key = ObjectMotionKey(object_label, frame - 1);

      gtsam::Pose3 previous_object_pose;
      CHECK(safeGetKey(object_centric_values, object_centric_previous_object_pose_key, &previous_object_pose))
          << "previous object pose key " << ObjectCentricKeyFormatter(object_centric_previous_object_pose_key);

      gtsam::Key world_centric_motion_key = world_centric_object_key_map.at(frame).at(object_label);
      gtsam::Pose3 object_motion;  // from previous to current
      CHECK(safeGetKey(world_centric_values, world_centric_motion_key, &object_motion))
          << " failed to get world_centric motion key between frames " << frame - 1 << " and " << frame << " object id "
          << object_label << " " << world_centric_motion_key;
      CHECK(!object_centric_values.exists(object_centric_previous_motion_key));

      // this motion is in world_centric_sam so add mapping
      world_centric_slam_to_dyna_keys.addMapping(world_centric_motion_key, object_centric_previous_motion_key);
      // world_centric_slam_to_dyna_keys[world_centric_motion_key] = object_centric_previous_motion_key;

      initaliseObjectMotionValue(object_centric_values, object_centric_previous_motion_key, object_motion);

      // if not added to the values yet, do so
      if (!object_centric_values.exists(object_centric_object_pose_key))
      {
        gtsam::Pose3 current_object_pose = createPropogatedObjectPose(
            object_motion, previous_object_pose,
            calculateCentroid(object_features, camera_pose, tracklets, object_id_manager, object_label),
            gt_object_result->object_pose_second_world_gt);

        // int gt_object_label = object_id_manager.getGtLabel(object_label);

        // const ObjectMotionResults& gt_object_motion_results = gt_objects.at(gt_object_label);
        // auto it = std::find_if(gt_object_motion_results.begin(), gt_object_motion_results.end(),
        //                       FindFrameIdObjectMotionResult(frame - 1));
        // CHECK(it != gt_object_motion_results.end());

        // const ObjectMotionResult::shared_ptr& gt_object_result = *it;

        // initalise pose only
        //  gtsam::Pose3 initial_object_pose = calculateCentroid(
        //        object_features, camera_pose, tracklets, object_id_manager, object_label);
        // object_centric_values.insert(object_centric_object_pose_key, gt_object_result->object_pose_second_world_gt);

        object_centric_values.insert(object_centric_object_pose_key, current_object_pose);
      }

      for (const gtsam::Key& key : object_features)
      {
        const Feature::shared_ptr& feature = tracklets.atFeature(key);
        DynamicFeature::shared_ptr dynamic_feature = std::dynamic_pointer_cast<DynamicFeature>(feature);
        CHECK(dynamic_feature);
        // if (dynamic_feature->isLastInTracklet())
        // {
        //   CHECK(graph_deconstruction.tracklets.trackletFromKey(key).index(feature) ==
        //         graph_deconstruction.tracklets.trackletFromKey(key).size() - 1);
        //   continue;
        // }

        if (FLAGS_use_object_centric_motion_factor)
        {
          gtsam::Key object_centric_dynamic_point_key = M(feature->tracklet_id);
          object_centric::ObjectCentricLandmarkMotionFactor::shared_ptr landmark_factor =
              boost::make_shared<object_centric::ObjectCentricLandmarkMotionFactor>(
                  object_centric_previous_motion_key, object_centric_previous_object_pose_key, object_centric_object_pose_key,
                  object_centric_dynamic_point_key, landmark_tenraryfactor_noisemodel);
          object_centric_graph.add(landmark_factor);
        }
      }

      if (FLAGS_use_object_odometry_factor)
      {
        auto object_odom_factor = boost::make_shared<object_centric::ObjectOdometryFactor>(
            object_centric_previous_motion_key, object_centric_previous_object_pose_key, object_centric_object_pose_key,
            object_motion_noisemodel);
        object_centric_graph.add(object_odom_factor);
      }

      const gtsam::Key trace_frame = frame - 2;
      const gtsam::Key object_centric_trace_motion_key = ObjectMotionKey(object_label, trace_frame);
      if (object_centric_values.exists(object_centric_trace_motion_key))
      {
        LOG(INFO) << "Smoothing factor between " << ObjectCentricKeyFormatter(object_centric_trace_motion_key) << " and "
                  << ObjectCentricKeyFormatter(object_centric_previous_motion_key);
        gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr smoothing_factor =
            boost::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                object_centric_trace_motion_key, object_centric_previous_motion_key, gtsam::Pose3::Identity(),
                smoothingfactor_noisemodel);
        object_centric_graph.add(smoothing_factor);
      }
    }
  }
}

bool addDynamicPoint(const Feature::shared_ptr feature, const gtsam::Values& world_centric_values,
                     gtsam::Values& object_centric_values, gtsam::NonlinearFactorGraph& object_centric_graph, int object_id,
                     WorldObjectCentricLikeMapping& world_centric_slam_to_dyna_keys, const int current_frame_id,
                     const gtsam::Key cam_pose_key,  // object_centric like (ie gtsam style)
                     gtsam::SharedNoiseModel noise_model)
{
  auto dynamic_feature = std::dynamic_pointer_cast<DynamicFeature>(feature);
  CHECK(dynamic_feature);

  const gtsam::Key& world_centric_point_key = feature->key;

  gtsam::Key object_centric_dynamic_point_key = M(feature->tracklet_id);
  gtsam::Key object_centric_cam_key = cam_pose_key;
  gtsam::Point3 dynamic_point_world = world_centric_values.at<gtsam::Point3>(world_centric_point_key);
  gtsam::Key object_centric_object_pose_key = ObjectPoseKey(object_id, current_frame_id);

  gtsam::Pose3 object_pose_world;
  // CHECK(safeGetKey(object_centric_values, object_centric_object_pose_key, &object_pose_world));
  if (!safeGetKey(object_centric_values, object_centric_object_pose_key, &object_pose_world))
  {
    return false;  // return?
  }
  bool is_new = false;

  // if we have not seen this point before, initalise it
  if (!object_centric_values.exists(object_centric_dynamic_point_key))
  {
    const gtsam::Point3 dynamic_point_world = world_centric_values.at<gtsam::Point3>(world_centric_point_key);
    gtsam::Point3 point_object = object_pose_world.inverse() * dynamic_point_world;
    object_centric_values.insert(object_centric_dynamic_point_key, point_object);
    world_centric_slam_to_dyna_keys.addMapping(world_centric_point_key, object_centric_dynamic_point_key);
    // world_centric_slam_to_dyna_keys[world_centric_point_key] = object_centric_dynamic_point_key;
    is_new = true;
  }
  gtsam::Point3 point_camera = dynamic_feature->measurement;
  // maybe need a different noise model
  object_centric::ObjectCentricPoint3DFactor::shared_ptr object_centric_point_factor =
      boost::make_shared<object_centric::ObjectCentricPoint3DFactor>(object_centric_cam_key, object_centric_object_pose_key,
                                                                object_centric_dynamic_point_key, point_camera, noise_model);
  object_centric_graph.add(object_centric_point_factor);
  return is_new;
}

void initalisePastDynamicPoints(int current_frame, int first_frame, int object_label,
                                const ObjectManager& object_manager_object_centric, const TrackletMap& tracklets,
                                const gtsam::Values& world_centric_values, gtsam::Values& object_centric_values,
                                gtsam::NonlinearFactorGraph& object_centric_graph, WorldObjectCentricLikeMapping& world_centric_slam_to_dyna_keys,
                                gtsam::SharedNoiseModel dynamic_point3dfactor_noisemodel)
{
  for (int frame = first_frame; frame < current_frame; frame++)
  {
    const ObjectObservations& feature_keys_per_label = object_manager_object_centric.at(frame);
    CHECK(feature_keys_per_label.objectObserved(object_label))
        << "Object " << object_label << " was not seen at " << frame;
    const gtsam::KeyVector object_features = feature_keys_per_label.at(object_label);
    const gtsam::Key cam_pose_key = X(frame);

    for (const gtsam::Key feature_key : object_features)
    {
      const Feature::shared_ptr& feature = tracklets.atFeature(feature_key);
      auto dynamic_feature = CHECK_NOTNULL(std::dynamic_pointer_cast<DynamicFeature>(feature));

      gtsam::Key world_centric_cam_key = dynamic_feature->camera_pose_key;
      gtsam::Symbol object_centric_cam_sym(world_centric_slam_to_dyna_keys.getObjectCentricKey(world_centric_cam_key));
      int observing_camera_pose_frame = object_centric_cam_sym.index();
      CHECK(observing_camera_pose_frame == frame);
      const gtsam::Key cam_pose_key = X(frame);
      CHECK(cam_pose_key == object_centric_cam_sym);

      // CHECK(object_id_manager.hasDynaLikeID(previous_dynamic_feature->object_label.get()));
      // CHECK_EQ(object_id_manager.getObjectCentricID(previous_dynamic_feature->object_label.get(), false),
      //          object_label);
      addDynamicPoint(dynamic_feature, world_centric_values, object_centric_values, object_centric_graph, object_label,
                      world_centric_slam_to_dyna_keys, frame, cam_pose_key, dynamic_point3dfactor_noisemodel);
    }
  }
}

gtsam::io::ValuesFactorGraphPair constructDynaSLAMLikeGraph(const GraphDeconstruction& graph_deconstruction,
                                                            const ObjectLabelResultMap& gt_objects,
                                                            const gtsam::NonlinearFactorGraph& world_centric_graph,
                                                            const gtsam::Values& world_centric_values,
                                                            ObjectManager& object_manager_object_centric,
                                                            object_centric::ObjectIDManager& object_id_manager,
                                                            WorldObjectCentricLikeMapping& world_centric_slam_to_dyna_keys)
{
  gtsam::NonlinearFactorGraph object_centric_graph;
  gtsam::Values object_centric_values;

  // mapping of world_centricslam keys to object_centric keys (should be a 1 to 1 of the keys in world_centric but the object_centric graph
  // will contain more keys)
  world_centric_slam_to_dyna_keys.clear();

  ObjectManager object_manager;  // uses the original (gt) object labelling and the original world_centric keys

  LOG(INFO) << "Creating DynaSLAM-like graph";

  FrameToKeyMap cam_pose_to_feature_key = graph_deconstruction.tracklets.reorderToPerFrame();

  // all the factors have the same covariance from world_centric so we extract the covariances from their noise models to copy
  // over
  gtsam::SharedNoiseModel static_point3dfactor_noisemodel;
  gtsam::SharedNoiseModel dynamic_point3dfactor_noisemodel;
  gtsam::SharedNoiseModel landmark_tenraryfactor_noisemodel;
  gtsam::SharedNoiseModel smoothingfactor_noisemodel;
  gtsam::SharedNoiseModel odometryfactor_noisemodel;
  gtsam::SharedNoiseModel object_motion_noisemodel;

  {
    const size_t factor_idx = graph_deconstruction.static_3d_point_idx.at(0);
    gtsam::NonlinearFactor::shared_ptr factor = world_centric_graph.at(factor_idx);
    const auto& point3d_factor = boost::dynamic_pointer_cast<dsc::Point3DFactor>(factor);
    CHECK(point3d_factor);
    // need to copy??
    static_point3dfactor_noisemodel = createFromworld_centricNoiseModel(point3d_factor->noiseModel());
  }

  {
    const size_t factor_idx = graph_deconstruction.dynamic_3d_point_idx.at(0);
    gtsam::NonlinearFactor::shared_ptr factor = world_centric_graph.at(factor_idx);
    const auto& point3d_factor = boost::dynamic_pointer_cast<dsc::Point3DFactor>(factor);
    CHECK(point3d_factor);
    // need to copy??

    // gtsam::SharedNoiseModel model = point3d_factor->noiseModel();
    // auto noise_model = gtsam::noiseModel::Isotropic::Sigma(3, FLAGS_object_centric_point_factor_sigma);
    // auto huber =
    //     gtsam::noiseModel::mEstimator::Huber::Create(0.01,
    //     gtsam::noiseModel::mEstimator::Base::ReweightScheme::Block);
    // gtsam::noiseModel::Base::shared_ptr robust_noise = gtsam::noiseModel::Robust::Create(huber, noise_model);
    // dynamic_point3dfactor_noisemodel = robust_noise;
    // gtsam::noiseModel::Base::shared_ptr noise_model = gtsam::noiseModel::Isotropic::Sigmas(model->sigmas());
    dynamic_point3dfactor_noisemodel = createFromworld_centricNoiseModel(point3d_factor->noiseModel());
    // dynamic_point3dfactor_noisemodel = noise_model;
  }

  {
    const size_t factor_idx = graph_deconstruction.landmark_ternary_factor_idx.at(0);
    gtsam::NonlinearFactor::shared_ptr factor = world_centric_graph.at(factor_idx);
    const auto& landmark_tenrary_factor = boost::dynamic_pointer_cast<dsc::LandmarkMotionTernaryFactor>(factor);
    CHECK(landmark_tenrary_factor);
    // need to copy??
    // landmark_tenraryfactor_noisemodel = landmark_tenrary_factor->noiseModel();
    // gtsam::SharedNoiseModel model = landmark_tenrary_factor->noiseModel();
    // gtsam::noiseModel::Base::shared_ptr noise_model = gtsam::noiseModel::Isotropic::Sigmas(model->sigmas());
    // landmark_tenraryfactor_noisemodel = createFromworld_centricNoiseModel(landmark_tenrary_factor->noiseModel());
    auto noise_model = gtsam::noiseModel::Isotropic::Sigma(3, FLAGS_object_centric_motion_factor_sigma);
    auto huber =
        gtsam::noiseModel::mEstimator::Huber::Create(0.001, gtsam::noiseModel::mEstimator::Base::ReweightScheme::Block);
    landmark_tenraryfactor_noisemodel = gtsam::noiseModel::Robust::Create(huber, noise_model);
  }

  {
    const size_t factor_idx = graph_deconstruction.smoothing_factor_idx.at(0);
    gtsam::NonlinearFactor::shared_ptr factor = world_centric_graph.at(factor_idx);
    const auto& between_factor = boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(factor);
    CHECK(between_factor);
    // need to copy??
    smoothingfactor_noisemodel = createFromworld_centricNoiseModel(between_factor->noiseModel());
  }

  {
    const size_t factor_idx = graph_deconstruction.between_factor_idx.at(0);
    gtsam::NonlinearFactor::shared_ptr factor = world_centric_graph.at(factor_idx);
    const auto& between_factor = boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(factor);
    CHECK(between_factor);
    // need to copy??
    odometryfactor_noisemodel = createFromworld_centricNoiseModel(between_factor->noiseModel());
  }

  object_motion_noisemodel = gtsam::noiseModel::Isotropic::Sigma(6, FLAGS_object_odometry_sigma);

  CHECK_NOTNULL(static_point3dfactor_noisemodel);
  CHECK_NOTNULL(dynamic_point3dfactor_noisemodel);
  CHECK_NOTNULL(landmark_tenraryfactor_noisemodel);
  CHECK_NOTNULL(smoothingfactor_noisemodel);
  CHECK_NOTNULL(odometryfactor_noisemodel);

  static_point3dfactor_noisemodel->print("Static point factor ");
  dynamic_point3dfactor_noisemodel->print("Dynamic point factor ");
  landmark_tenraryfactor_noisemodel->print("Landmark ternary factor ");
  smoothingfactor_noisemodel->print("Smoothing factor ");
  odometryfactor_noisemodel->print("Odom factor ");

  // throw std::runtime_error("Test");

  // has to be at least 3 (this avoid the problem of objects that only appear once).
  //  The logic to deal with them is quite complicated so we just just start from the 2nd observation
  // const int kMinObjectObservations = 5;
  // const int kMinObjectObservations = 15;
  const size_t max_features_for_testing = 10;

  const int kMinObjectObservations = std::max(FLAGS_min_object_observations, 3);

  // start at 1 to match world_centric formatting
  size_t current_frame_id = 1;

  std::map<int, gtsam::Pose3> object_pose_initalisation;

  // frame -> object id (object_centric like) -> world_centric key for object motion
  std::map<int, std::map<int, gtsam::Key>> world_centric_object_key_map;

  // start with camera poses
  const gtsam::KeySet& camera_pose_keys = graph_deconstruction.camera_pose_keys;
  for (const gtsam::Key& cam_pose_key : camera_pose_keys)
  {
    CHECK(cam_pose_to_feature_key.find(cam_pose_key) != cam_pose_to_feature_key.end());

    // add camera vertex
    gtsam::Pose3 cam_pose = world_centric_values.at<gtsam::Pose3>(cam_pose_key);

    gtsam::Key dyna_cam_key = X(current_frame_id);
    object_centric_values.insert(dyna_cam_key, cam_pose);
    world_centric_slam_to_dyna_keys.addMapping(cam_pose_key, dyna_cam_key);
    // world_centric_slam_to_dyna_keys[cam_pose_key] = dyna_cam_key;

    // assume camera keys start at 1
    if (cam_pose_key == 1)
    {
      gtsam::PriorFactor<gtsam::Pose3>::shared_ptr original_prior_factor_ptr =
          boost::dynamic_pointer_cast<gtsam::PriorFactor<gtsam::Pose3>>(world_centric_graph.at(0));
      CHECK(original_prior_factor_ptr);
      auto prior_factor = original_prior_factor_ptr->rekey({ dyna_cam_key });
      object_centric_graph.add(prior_factor);
      LOG(INFO) << "Adding pose prior factor";
    }
    else
    {
      gtsam::Key previous_cam_key = X(current_frame_id - 1);
      gtsam::Pose3 previous_pose = object_centric_values.at<gtsam::Pose3>(previous_cam_key);
      gtsam::Pose3 odom = previous_pose.inverse() * cam_pose;
      object_centric_graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(previous_cam_key, dyna_cam_key, odom,
                                                                         odometryfactor_noisemodel);
    }

    // all features observed at this idx
    const gtsam::KeyVector& observed_features = cam_pose_to_feature_key[cam_pose_key];

    // map of object label (including background) to seen keypoints in this frame using the original world_centric keys
    // first collcect all features and sort into object labels: 0 (static), 1, 2... (dynamic)
    ObjectObservations label_to_keys(current_frame_id);

    size_t static_feature_count = 0;
    size_t dynamic_feature_count = 0;

    for (const auto& key : observed_features)
    {
      const Feature::shared_ptr feature = graph_deconstruction.tracklets.atFeature(key);
      const Tracklet& tracklet = graph_deconstruction.tracklets.at(feature->tracklet_id);
      // if(tracklet.size() < 4) {
      //   continue;
      // }
      const gtsam::Key& point_key_world_centric = key;
      bool is_dynamic = feature->isDynamic();

      if (is_dynamic /*&& dynamic_feature_count < max_features_for_testing*/)
      {
        auto dynamic_feature = std::dynamic_pointer_cast<DynamicFeature>(feature);
        CHECK(dynamic_feature);
        label_to_keys.addObjectObservation(*dynamic_feature->object_label, point_key_world_centric);  // gt label
        dynamic_feature_count++;
      }

      if (!is_dynamic /*&& static_feature_count < max_features_for_testing*/)
      {
        auto static_feature = std::dynamic_pointer_cast<StaticFeature>(feature);
        // if this fails we have a feature that is dynamic but the object label is no longer initalised due to a failed
        // ground truth observation in this case, throw it out
        if (!static_feature)
        {
          // LOG(ERROR) << "missing feature "
          continue;
        }
        CHECK(static_feature);
        CHECK(static_feature->observingCameraPose(cam_pose_key));
        label_to_keys.addObjectObservation(0, point_key_world_centric);  // gt label
        static_feature_count++;
      }
    }

    object_manager.safeAdd(label_to_keys);
    // ObjectObservations object_centric_label_to_keys(label_to_keys.FrameId());
    ObjectObservations object_centric_label_to_keys(current_frame_id);
    std::map<int, gtsam::Key> object_ids_to_world_centric_motion_keys;  // new object ids to motion keys

    // index takes you to the current frame
    world_centric_object_key_map[current_frame_id] = std::map<int, gtsam::Key>();
    // LOG(INFO) << "For frame " << current_frame_id << " features " << observed_features.size()
    //           << "(static:" << static_feature_count << "/dynamic: " << dynamic_feature_count << ") object seen: ";
    // go through each object to initalise object pose, motion,  ObjectCentricLandmarkMotionFactor and smoothing factor
    for (const auto& it : label_to_keys)
    {
      // LOG(INFO) << "gt label=" << it.first << "(" << it.second.size() << ") ";

      const int gt_object_label = it.first;
      const gtsam::KeyVector& feature_keys_per_object = it.second;
      // ignore if background
      if (gt_object_label == 0)
      {
        object_centric_label_to_keys.addObjectObservations(0, feature_keys_per_object);
        continue;
      }

      // used to sanity check that all the keys on a single object have the same object motion key
      std::set<gtsam::Key> object_motion_key;
      std::set<int> object_label_set;
      std::vector<gtsam::Key> object_motion_key_vec;
      // in the case that all the features are end features (in a tracklet) the count should then
      // be the length of it.second. In any other case it just means that inidivdual tracklets are dropped in this frame
      // which is fine but they just wont have a motion_key attached to them so we handle that case with the
      // lastInTrackletCheck
      int contained_last_feature_count = 0;

      for (const gtsam::Key& key : feature_keys_per_object)
      {
        const Feature::shared_ptr& f = graph_deconstruction.tracklets.atFeature(key);
        DynamicFeature::shared_ptr dynamic_feature = std::dynamic_pointer_cast<DynamicFeature>(f);
        CHECK(dynamic_feature);

        object_label_set.insert(*dynamic_feature->object_label);

        if (dynamic_feature->isLastInTracklet())
        {
          CHECK(graph_deconstruction.tracklets.trackletFromKey(key).index(f) ==
                graph_deconstruction.tracklets.trackletFromKey(key).size() - 1);
          contained_last_feature_count++;
          continue;
        }
        object_motion_key.insert(*dynamic_feature->motion_key);
        object_motion_key_vec.push_back(*dynamic_feature->motion_key);
      }

      // indicates that tracking was lost in the world_centric system
      bool world_centric_tracking_lost = object_motion_key.empty();
      int object_label = object_id_manager.getObjectCentricID(gt_object_label, world_centric_tracking_lost);
      CHECK_EQ(object_id_manager.getGtLabel(object_label), gt_object_label);
      // bool world_centric_tracking_lost = object_motion_key.size() > 1;

      // should only be empty in the case where the feature is the last feature in the tracklet so we dont have a motion
      // key
      if (!world_centric_tracking_lost)
      {
        gtsam::Key object_motion_key_world_centric = *object_motion_key.begin();

        CHECK_LT(contained_last_feature_count,
                 feature_keys_per_object.size());  // if tracking not lost, we still should have some features
        CHECK_EQ(object_label_set.size(), 1u) << "Not all the features for object " << object_label
                                              << " had the same object label " << container_to_string(object_label_set);
        // CHECK_EQ(object_motion_key.size(), 1u)
        //     << "Not all the features for object " << object_label << " had the same object motion key "
        //     << container_to_string(object_motion_key);
        if (object_motion_key.size() != 1u)
        {
          // use the one that appeared the most
          auto result = getMostFrequentElement(object_motion_key_vec);
          object_motion_key_world_centric = result.first;

          LOG(WARNING) << "Not all the features for object " << object_label << " had the same object motion key "
                       << container_to_string(object_motion_key) << ", using " << object_motion_key_world_centric
                       << " which appeared " << result.second << " times";

          //
        }

        // CHECK_EQ(object_label, object_id_manager.getGtLabel(*object_label_set.begin()));

        object_centric_label_to_keys.addObjectObservations(object_label, feature_keys_per_object);
        object_ids_to_world_centric_motion_keys[object_label] = object_motion_key_world_centric;
        world_centric_slam_to_dyna_keys.addMapping(object_motion_key_world_centric, ObjectMotionKey(object_label, current_frame_id));
        // world_centric_slam_to_dyna_keys[object_motion_key_world_centric] = ObjectMotionKey(object_label, current_frame_id);

        std::map<int, gtsam::Key>& object_id_world_centric_motion_keys = world_centric_object_key_map.at(current_frame_id);
        object_id_world_centric_motion_keys[object_label] = object_motion_key_world_centric;
      }
      else
      {
        // CHECK_EQ(contained_last_feature_count, feature_keys_per_object.size());
        // CHECK_EQ(object_motion_key.size(), 0u);

        LOG(WARNING) << "skipping gt object " << gt_object_label << " at frame " << current_frame_id;
        // this shoudl also only happen at the ennd so lets guarantee that we have already initalised the motion of this
        // object
        // CHECK(object_pose_initalisation.find(object_label) != object_pose_initalisation.end());
        // int old_object_label =  object_id_manager.getObjectCentricID(gt_object_label, false);
        // gtsam::Key object_motion_key_world_centric = *object_motion_key.begin();
        // object_centric_label_to_keys.addObjectObservations(old_object_label, feature_keys_per_object);
        // object_ids_to_world_centric_motion_keys[old_object_label] = object_motion_key_world_centric;
        // world_centric_slam_to_dyna_keys[object_motion_key_world_centric] = ObjectMotionKey(old_object_label, current_frame_id);

        // std::map<int, gtsam::Key>& object_id_world_centric_motion_keys = world_centric_object_key_map.at(current_frame_id);
        // object_id_world_centric_motion_keys[old_object_label] = object_motion_key_world_centric;
      }
    }

    object_manager_object_centric.safeAdd(object_centric_label_to_keys);

    LOG(INFO) << "Starting frame " << current_frame_id << " with objects "
              << container_to_string(object_centric_label_to_keys.ObservedObjects());

    for (const auto& it : object_centric_label_to_keys)
    {
      const int object_label = it.first;  // object_centric object label

      if (object_label == 0)
      {
        continue;
      }

      int num_observations = object_manager_object_centric.numObservations(object_label);
      LOG(INFO) << "assigned label=" << object_label << "(gt label=" << object_id_manager.getGtLabel(object_label)
                << "), frame=" << current_frame_id << " num observations= " << num_observations;

      const gtsam::KeyVector& feature_keys_per_object = it.second;
      CHECK(num_observations != -1);

      if (num_observations < kMinObjectObservations)
      {
        continue;
      }
      else if (num_observations == kMinObjectObservations)
      {
        initalisePastObjects(gt_objects, current_frame_id, current_frame_id - kMinObjectObservations + 1,
                             object_manager_object_centric, object_id_manager, graph_deconstruction.tracklets,
                             world_centric_object_key_map, object_label, world_centric_values, world_centric_slam_to_dyna_keys, object_centric_values,
                             object_centric_graph, object_motion_noisemodel, smoothingfactor_noisemodel,
                             landmark_tenraryfactor_noisemodel);
      }

      // LOG(INFO) << "newly assignmed label=" << object_label <
      // object that we have already seen before
      const gtsam::Key object_centric_object_pose_key = ObjectPoseKey(object_label, current_frame_id);
      // LOG(INFO) << "tracking object pose " << ObjectCentricKeyFormatter(object_centric_object_pose_key);

      // we start the current_frame_id at 1 so we should never get here on the first frame
      // this might not be the case if we drop and object and reclassify it?
      CHECK_GT(current_frame_id, 1);
      int previous_seen_frame = current_frame_id - 1;  // assume all frames when object tracking are consequative. If
                                                       // not, the object should be reclassified
      // use the motion labelled form the previous frame as we notate a motion from the previous to the current frame
      // so this motion is added in the current frame
      gtsam::Key object_centric_previous_motion_key = ObjectMotionKey(object_label, previous_seen_frame);

      gtsam::Key previous_object_pose_key = ObjectPoseKey(object_label, previous_seen_frame);
      gtsam::Pose3 previous_object_pose;
      if (!safeGetKey(object_centric_values, previous_object_pose_key, &previous_object_pose))
      {
        // may happen due to occlusions?
        // make new id for this object and continue
        int new_label = object_id_manager.getObjectCentricID(object_id_manager.getGtLabel(object_label), true);
        object_manager_object_centric.at(current_frame_id).remap(object_label, new_label);
        LOG(WARNING) << "previous object pose key not found " << ObjectCentricKeyFormatter(previous_object_pose_key)
                     << " remapping object id " << object_label << " -> " << new_label
                     << " for gt label=" << object_id_manager.getGtLabel(object_label) << " at frame "
                     << current_frame_id;
        continue;
      }

      // gtsam::Key world_centric_motion_key = object_ids_to_world_centric_motion_keys[object_label];
      gtsam::Key world_centric_motion_key = world_centric_object_key_map[current_frame_id][object_label];
      // check motion from previous to current frame exists in world_centric-sam
      gtsam::Pose3 object_motion;  // from previous to current
      CHECK(safeGetKey(world_centric_values, world_centric_motion_key, &object_motion))
          << " failed to get world_centric motion key between frames " << previous_seen_frame << " and " << current_frame_id
          << " object id " << object_label << " " << world_centric_motion_key;
      CHECK(!object_centric_values.exists(object_centric_previous_motion_key));
      initaliseObjectMotionValue(object_centric_values, object_centric_previous_motion_key, object_motion);

      // this motion is in world_centric_sam so add mapping
      world_centric_slam_to_dyna_keys.addMapping(world_centric_motion_key, object_centric_previous_motion_key);
      // world_centric_slam_to_dyna_keys[world_centric_motion_key] = object_centric_previous_motion_key;

      // if not added to the values yet, do so
      if (!object_centric_values.exists(object_centric_object_pose_key))
      {
        // initalise pose from object motion
        // TODO: multi-pre-multiplication
        // gtsam::Pose3 current_object_pose_propogated = object_motion * previous_object_pose;
        gtsam::Pose3 camera_pose = object_centric_values.at<gtsam::Pose3>(X(current_frame_id));

        const int gt_object_label = object_id_manager.getGtLabel(object_label);
        const ObjectMotionResults& gt_object_motion_results = gt_objects.at(gt_object_label);
        auto it = std::find_if(gt_object_motion_results.begin(), gt_object_motion_results.end(),
                               FindFrameIdObjectMotionResult(current_frame_id - 1));

        boost::optional<gtsam::Pose3> gt_pose;
        if (it != gt_object_motion_results.end())
        {
          const ObjectMotionResult::shared_ptr& gt_object_result = *it;
          gt_pose = gt_object_result->object_pose_second_world_gt;
        }

        gtsam::Pose3 current_object_pose = createPropogatedObjectPose(
            object_motion, previous_object_pose,
            calculateCentroid(feature_keys_per_object, camera_pose, graph_deconstruction.tracklets, object_id_manager,
                              object_label),
            gt_pose);

        // int gt_object_label = object_id_manager.getGtLabel(object_label);

        // const ObjectMotionResults& gt_object_motion_results = gt_objects.at(gt_object_label);
        // auto it = std::find_if(gt_object_motion_results.begin(), gt_object_motion_results.end(),
        //                       FindFrameIdObjectMotionResult(current_frame_id - 1));
        // CHECK(it != gt_object_motion_results.end());

        // const ObjectMotionResult::shared_ptr& gt_object_result = *it;

        object_centric_values.insert(object_centric_object_pose_key, current_object_pose);
        // object_centric_values.insert(object_centric_object_pose_key, gt_object_result->object_pose_second_world_gt);
        // object_centric_graph.addPrior<gtsam::Pose3>(
        //     object_centric_object_pose_key, current_object_pose_propogated,
        //     gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 0.5, 0.5, 0.5, 0.4, 0.4, 0.4).finished()));
      }

      if (FLAGS_use_object_odometry_factor)
      {
        LOG(INFO) << "Using OOF";
        auto object_odom_factor =
            boost::make_shared<object_centric::ObjectOdometryFactor>(object_centric_previous_motion_key, previous_object_pose_key,
                                                                object_centric_object_pose_key, object_motion_noisemodel);
        object_centric_graph.add(object_odom_factor);
      }

      // // //add smoothing factor -> should be from two times ago
      const gtsam::Key trace_frame = current_frame_id - 2;
      const gtsam::Key object_centric_trace_motion_key = ObjectMotionKey(object_label, trace_frame);
      if (object_centric_values.exists(object_centric_trace_motion_key))
      {
        LOG(INFO) << "Smoothing factor between " << ObjectCentricKeyFormatter(object_centric_trace_motion_key) << " and "
                  << ObjectCentricKeyFormatter(object_centric_previous_motion_key);
        gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr smoothing_factor =
            boost::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(
                object_centric_trace_motion_key, object_centric_previous_motion_key, gtsam::Pose3::Identity(),
                smoothingfactor_noisemodel);
        object_centric_graph.add(smoothing_factor);
      }

      if (FLAGS_use_object_centric_motion_factor)
      {
        LOG(INFO) << "Using OCMF";
        // add ObjectCentricLandmarkMotionFactor for each point on the object
        for (const gtsam::Key& key : feature_keys_per_object)
        {
          const Feature::shared_ptr& feature = graph_deconstruction.tracklets.atFeature(key);
          DynamicFeature::shared_ptr dynamic_feature = std::dynamic_pointer_cast<DynamicFeature>(feature);
          CHECK(dynamic_feature);
          // if (dynamic_feature->isLastInTracklet())
          // {
          //   CHECK(graph_deconstruction.tracklets.trackletFromKey(key).index(feature) ==
          //         graph_deconstruction.tracklets.trackletFromKey(key).size() - 1);
          //   continue;
          // }
          gtsam::Key object_centric_dynamic_point_key = M(feature->tracklet_id);
          object_centric::ObjectCentricLandmarkMotionFactor::shared_ptr landmark_factor =
              boost::make_shared<object_centric::ObjectCentricLandmarkMotionFactor>(
                  object_centric_previous_motion_key, previous_object_pose_key, object_centric_object_pose_key,
                  object_centric_dynamic_point_key, landmark_tenraryfactor_noisemodel);
          object_centric_graph.add(landmark_factor);
        }
      }
    }

    // now we have object so we can do the static and dynamic object points
    for (const auto& it : object_centric_label_to_keys)
    {
      const int object_label = it.first;
      const gtsam::KeyVector& feature_keys_per_label = it.second;

      if (object_label == 0)
      {
        // treat as static points
        for (const gtsam::Key& key : feature_keys_per_label)
        {
          const Feature::shared_ptr& feature = graph_deconstruction.tracklets.atFeature(key);
          const gtsam::Key& world_centric_point_key = key;

          gtsam::Key object_centric_cam_key = world_centric_slam_to_dyna_keys.getObjectCentricKey(cam_pose_key);
          // gtsam::Key object_centric_cam_key = world_centric_slam_to_dyna_keys[cam_pose_key];
          CHECK(object_centric_values.exists(object_centric_cam_key));

          auto static_feature = std::dynamic_pointer_cast<StaticFeature>(feature);
          CHECK(static_feature);

          CHECK(static_feature->observingCameraPose(cam_pose_key));
          gtsam::Key dyna_static_point_key = P(feature->tracklet_id);

          // if feature is static and we have not seen before, we add to the values set
          if (!object_centric_values.exists(dyna_static_point_key))
          {
            // get value using the original point key
            gtsam::Point3 static_point_world = world_centric_values.at<gtsam::Point3>(world_centric_point_key);
            object_centric_values.insert(dyna_static_point_key, static_point_world);
            world_centric_slam_to_dyna_keys.addMapping(world_centric_point_key, dyna_static_point_key);
            // world_centric_slam_to_dyna_keys[world_centric_point_key] = dyna_static_point_key;
          }

          // add point3d factor for this observation
          gtsam::Point3 measurement;
          CHECK(static_feature->getMeasurement(measurement, cam_pose_key));
          dsc::Point3DFactor::shared_ptr point_factor = boost::make_shared<dsc::Point3DFactor>(
              object_centric_cam_key, dyna_static_point_key, measurement, static_point3dfactor_noisemodel);
          point_factor->is_dynamic_ = false;
          object_centric_graph.add(point_factor);
        }
      }
      else
      {
        int num_observations = object_manager_object_centric.numObservations(object_label);
        CHECK(num_observations != -1);

        if (num_observations < kMinObjectObservations)
        {
          continue;
        }
        else if (num_observations == kMinObjectObservations)
        {
          initalisePastDynamicPoints(current_frame_id, current_frame_id - kMinObjectObservations + 1, object_label,
                                     object_manager_object_centric, graph_deconstruction.tracklets, world_centric_values,
                                     object_centric_values, object_centric_graph, world_centric_slam_to_dyna_keys,
                                     dynamic_point3dfactor_noisemodel);
        }

        size_t n_new_features = 0;
        for (const gtsam::Key& key : feature_keys_per_label)
        {
          const Feature::shared_ptr& feature = graph_deconstruction.tracklets.atFeature(key);
          const gtsam::Key& world_centric_point_key = key;

          gtsam::Key object_centric_cam_key = world_centric_slam_to_dyna_keys.getObjectCentricKey(cam_pose_key);
          // gtsam::Key object_centric_cam_key = world_centric_slam_to_dyna_keys[cam_pose_key];
          CHECK(object_centric_values.exists(object_centric_cam_key));

          auto dynamic_feature = std::dynamic_pointer_cast<DynamicFeature>(feature);
          CHECK(dynamic_feature);

          gtsam::Key world_centric_cam_key = dynamic_feature->camera_pose_key;
          // gtsam::Symbol object_centric_cam_sym(world_centric_slam_to_dyna_keys[world_centric_cam_key]);
          gtsam::Symbol object_centric_cam_sym(world_centric_slam_to_dyna_keys.getObjectCentricKey(world_centric_cam_key));
          int observing_camera_pose = object_centric_cam_sym.index();
          CHECK(object_centric_cam_key == X(observing_camera_pose));
          bool is_new = addDynamicPoint(dynamic_feature, world_centric_values, object_centric_values, object_centric_graph, object_label,
                                        world_centric_slam_to_dyna_keys, current_frame_id, X(observing_camera_pose),
                                        dynamic_point3dfactor_noisemodel);
          if (is_new)
          {
            n_new_features++;
          }
        }
        LOG(INFO) << "number features=" << feature_keys_per_label.size() << " for object label=" << object_label
                  << " where # new=" << n_new_features;
      }
    }
    current_frame_id++;

    if (current_frame_id == 20)
    {
      object_centric_graph.saveGraph("/root/world_centric-SAM/object_centricgraph.dot", ObjectCentricKeyFormatter);
      // break;
    }
  }
  // object_centric_graph.saveGraph("/root/world_centric-SAM/object_centricgraph.dot", ObjectCentricKeyFormatter);

  return std::make_pair(object_centric_values, object_centric_graph);
}

// void saveDynaLikeCameraPoses(const CameraPoseResultMap& pose_results, const gtsam::Values& object_centric_values,
//                              const std::string& object_pose_file, const std::string& path)
// {
//   const std::string pose_path = path + object_pose_file;
//   std::ofstream pose_file;
//   pose_file.open(pose_path);

//   for (const auto& it : pose_results)
//   {
//     const int dyna_frame_id =
//         it.first + 1;  // object_centric starts frame id at 1 like world_centric but gt starts frame id at 0 (like a normal person)
//     gtsam::Symbol pose_sym(X(dyna_frame_id));

//     if (!object_centric_values.exists(pose_sym))
//     {
//       LOG(WARNING) << "Missing object_centric like camera pose at" << gtsam::DefaultKeyFormatter(pose_sym);
//       continue;
//     }
//     const gtsam::Pose3 object_centric_pose = object_centric_values.at<gtsam::Pose3>(pose_sym);
//     const gtsam::Pose3 gt_pose = it.second.ground_truth;
//     pose_file << "POSE3 " << object_centric_pose.x() << " " << object_centric_pose.y() << " " << object_centric_pose.z() << " "
//               << gt_pose.x() << " " << gt_pose.y() << " " << gt_pose.z() << "\n";
//   }
// }

std::pair<CameraPoseResultMap, ObjectLabelResultMap> constructDynaLikeResults(
    const CameraPoseResultMap& world_centric_camera_result_map, const ObjectLabelResultMap& world_centric_object_motion_result_map,
    const ObjectManager& object_manager_object_centric, const object_centric::ObjectIDManager& object_id_manager,
    const WorldObjectCentricLikeMapping& world_centric_slam_to_dyna_keys, const gtsam::Values& object_centric_initial_values,
    const gtsam::Values& object_centric_opt_values)
{
  ObjectLabelResultMap object_centric_object_results;
  CameraPoseResultMap object_centric_cam_results;
  for (const auto& it : object_manager_object_centric)
  {
    const int dyna_frame_id = it.first;
    const int frame_id =
        dyna_frame_id - 1;  // object_centric starts frame id at 1 like world_centric but gt starts frame id at 0 (like a normal person)
    const ObjectObservations& object_observations = it.second;

    const std::vector<int> seen_objects = object_observations.ObservedObjects();

    const CameraPoseResult& world_centric_cam_result = world_centric_camera_result_map.at(frame_id);
    CameraPoseResult object_centric_cam_result;
    object_centric_cam_result.ground_truth = world_centric_cam_result.ground_truth;
    object_centric_cam_result.initial_estimate = object_centric_initial_values.at<gtsam::Pose3>(X(dyna_frame_id));
    object_centric_cam_result.optimized_estimate = object_centric_opt_values.at<gtsam::Pose3>(X(dyna_frame_id));

    object_centric_cam_results[frame_id] = object_centric_cam_result;

    for (const int object_id : seen_objects)
    {
      // LOG(INFO) << " Saving result for object " << object_id << " frame id " << frame_id;
      if (object_id == 0)
      {
        continue;
      }

      if (object_centric_object_results.find(object_id) == object_centric_object_results.end())
      {
        object_centric_object_results[object_id] = ObjectMotionResults();
      }

      const int gt_object_id = object_id_manager.getGtLabel(object_id);
      const gtsam::Key object_centric_object_pose_key = ObjectPoseKey(object_id, dyna_frame_id);
      gtsam::Pose3 object_pose_initial, object_pose_refined;
      // CHECK(safeGetKey(object_centric_values, object_centric_object_pose_key, &object_pose))
      //   << "Missing object pose at key " << ObjectCentricKeyFormatter(object_centric_object_pose_key);
      if (!safeGetKey(object_centric_initial_values, object_centric_object_pose_key, &object_pose_initial))
      {
        continue;
        LOG(WARNING) << "Missing object pose at key " << ObjectCentricKeyFormatter(object_centric_object_pose_key);
      }

      safeGetKey(object_centric_opt_values, object_centric_object_pose_key, &object_pose_refined);

      // find in ObjectLabelResultMap which contains gt info
      const ObjectMotionResults& gt_object_motion_results = world_centric_object_motion_result_map.at(gt_object_id);
      auto it = std::find_if(gt_object_motion_results.begin(), gt_object_motion_results.end(),
                             FindFrameIdObjectMotionResult(frame_id));

      std::stringstream ss;
      ss << "failed to find object at frame " << frame_id << "\n";
      for (const auto& r : gt_object_motion_results)
      {
        ss << "[gt_object_id=" << r->object_label << " frame_id=" << r->frame_id << "]\n";
      }
      // CHECK(it != gt_object_motion_results.end()) << ss.str();
      if (it == gt_object_motion_results.end())
      {
        LOG(WARNING) << ss.str();
        continue;
      }

      const ObjectMotionResult::shared_ptr& gt_object_result = *it;
      CHECK_EQ(gt_object_result->frame_id, frame_id);
      CHECK_EQ(gt_object_result->object_label, gt_object_id);

      ObjectMotionResult::shared_ptr object_result = std::make_shared<ObjectMotionResult>(*gt_object_result);
      object_result->initial_estimate_L = object_pose_initial;
      object_result->optimized_estimate_L = object_pose_refined;

      // pose_file << "OBJECT_POSE3 " << frame_id << " " << object_id;
      // pose_file << " " << object_pose.x() << " " << object_pose.y() << " " << object_pose.z() << " "
      //           << gt_object_result->object_pose_first_world_gt.x() << " "
      //           << gt_object_result->object_pose_first_world_gt.y() << " "
      //           << gt_object_result->object_pose_first_world_gt.z() << "\n";

      if (dyna_frame_id > 1)
      {
        const gtsam::Key object_centric_object_motion_key = ObjectMotionKey(object_id, dyna_frame_id - 1);

        gtsam::Pose3 opt_estimate_motion_w, initial_estimate_motion_w;
        safeGetKey(object_centric_opt_values, object_centric_object_motion_key, &opt_estimate_motion_w);
        safeGetKey(object_centric_initial_values, object_centric_object_motion_key, &initial_estimate_motion_w);

        object_result->initial_estimate = initial_estimate_motion_w;
        object_result->optimized_estimate = opt_estimate_motion_w;
      }

      object_centric_object_results[gt_object_id].push_back(object_result);
    }
  }

  return std::make_pair(object_centric_cam_results, object_centric_object_results);
}
