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

#include "Common.h"
#include "GtsamIO.h"

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/inference/LabeledSymbol.h>

std::string getObjectCentricSystemName();

namespace object_centric
{

class ObjectCentricPoint3DFactor : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Point3>
{
private:
  gtsam::Point3 measured_;  // in camera coordinates

  /**
   * @brief Implementation of the residual fucntion e, without jacobians
   *
   * @param m_camera const gtsam::Point3& measurement in the camera frame of the point
   * @param X
   * @param L
   * @param m
   * @return gtsam::Vector
   */
  static gtsam::Vector3 calculateResidual(const gtsam::Point3& m_camera, const gtsam::Pose3& X, const gtsam::Pose3& L,
                                          const gtsam::Point3& m);

public:
  typedef boost::shared_ptr<ObjectCentricPoint3DFactor> shared_ptr;

  /**
   * @brief Construct a new Object Centric Point 3 D Factor object
   *
   * @param cameraPoseKey
   * @param objectPoseKey
   * @param objectPointKey connection to a point in object frame
   * @param measured point in camera frame
   * @param model
   */
  ObjectCentricPoint3DFactor(gtsam::Key cameraPoseKey, gtsam::Key objectPoseKey, gtsam::Key objectPointKey,
                             gtsam::Point3 measured, gtsam::SharedNoiseModel model);

  gtsam::NonlinearFactor::shared_ptr clone() const override
  {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new ObjectCentricPoint3DFactor(*this)));
  }

  /**
   * @brief
   *
   * @param X Camera pose in world frame
   * @param L Object pose in world frame
   * @param m object point in L (object) frame
   * @param J1
   * @param J2
   * @param J3
   * @return gtsam::Vector
   */
  gtsam::Vector evaluateError(const gtsam::Pose3& X, const gtsam::Pose3& L, const gtsam::Point3& m,
                              boost::optional<gtsam::Matrix&> J1 = boost::none,
                              boost::optional<gtsam::Matrix&> J2 = boost::none,
                              boost::optional<gtsam::Matrix&> J3 = boost::none) const override;

  inline const gtsam::Point3& measured() const
  {
    return measured_;
  }
};

/**
 * @brief Factor on two object poses at (t-1 and t) with the estimated motion H, with all variables in the world frame
 *
 */
class ObjectOdometryFactor : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>
{
private:
  static gtsam::Vector6 calculateResidual(const gtsam::Pose3& H_w, const gtsam::Pose3& L_previous_world,
                                          const gtsam::Pose3& L_current_world);

public:
  typedef boost::shared_ptr<ObjectOdometryFactor> shared_ptr;

  ObjectOdometryFactor(gtsam::Key motionKey, gtsam::Key previousObjectPoseKey, gtsam::Key currentObjectPoseKey,
                       gtsam::SharedNoiseModel model);

  gtsam::NonlinearFactor::shared_ptr clone() const override
  {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new ObjectOdometryFactor(*this)));
  }

  gtsam::Vector evaluateError(const gtsam::Pose3& H_w, const gtsam::Pose3& L_previous_world,
                              const gtsam::Pose3& L_current_world, boost::optional<gtsam::Matrix&> J1 = boost::none,
                              boost::optional<gtsam::Matrix&> J2 = boost::none,
                              boost::optional<gtsam::Matrix&> J3 = boost::none) const override;
};

/**
 * @brief Motion factor constraining motion estimation {t-1}H_t between L_{t-1}, L_t with  (all in world frame) of a
 * static point m_t in frame L
 *
 */
class ObjectCentricLandmarkMotionFactor
  : public gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3, gtsam::Point3>
{
private:
  static gtsam::Vector3 calculateResidual(const gtsam::Pose3& H_world, const gtsam::Pose3& L_previous_world,
                                          const gtsam::Pose3& L_current_world, const gtsam::Point3& point_L);

public:
  typedef boost::shared_ptr<ObjectCentricLandmarkMotionFactor> shared_ptr;

  ObjectCentricLandmarkMotionFactor(gtsam::Key motionKey, gtsam::Key previousObjectPoseKey,
                                    gtsam::Key currentObjectPoseKey, gtsam::Key objectPointKey,
                                    gtsam::SharedNoiseModel model);

  gtsam::NonlinearFactor::shared_ptr clone() const override
  {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new ObjectCentricLandmarkMotionFactor(*this)));
  }

  gtsam::Vector evaluateError(const gtsam::Pose3& H_w, const gtsam::Pose3& L_previous_world,
                              const gtsam::Pose3& L_current_world, const gtsam::Point3& point_L,
                              boost::optional<gtsam::Matrix&> J1 = boost::none,
                              boost::optional<gtsam::Matrix&> J2 = boost::none,
                              boost::optional<gtsam::Matrix&> J3 = boost::none,
                              boost::optional<gtsam::Matrix&> J4 = boost::none) const override;
};

class ObjectIDManager
{
public:
  int getObjectCentricID(int ground_truth_object_id, bool world_centric_lost_tracking);
  int getGtLabel(int object_centric_object_label) const;
  bool hasDynaLikeID(int ground_truth_object_id) const;

private:
  std::map<int, int> active_object_ids_;  //! Mapping from ground object id to "new" object id, taking int account when
                                          //! an object is new

  // object_centric associated object ids (as created from getObjectCentricID) to the gt.
  // there will be less gt's to object ids since one object (with one gt) can be (re)associated many times
  std::map<int, int> object_centric_object_ids_to_gt_;
  int max_object_id{ 1 };  //! starting object index
};

}  // namespace object_centric

class WorldObjectCentricLikeMapping
{
public:
  void clear()
  {
    world_to_object_.clear();
    object_to_world_.clear();
  }

  void addMapping(gtsam::Key world_centric, gtsam::Key object_centric)
  {
    world_to_object_.insert({ world_centric, object_centric });
    object_to_world_.insert({ object_centric, world_centric });
  }

  gtsam::Key getWorldCentricKey(gtsam::Key object_centric) const
  {
    return object_to_world_.at(object_centric);
  }

  gtsam::Key getObjectCentricKey(gtsam::Key world_centric) const
  {
    return world_to_object_.at(world_centric);
  }

private:
  std::map<gtsam::Key, gtsam::Key> world_to_object_;
  std::map<gtsam::Key, gtsam::Key> object_to_world_;
};

template <typename VALUE>
bool safeGetKey(const gtsam::Values& values, const gtsam::Key& key, VALUE* value)
{
  CHECK_NOTNULL(value);
  if (!values.exists(key))
  {
    return false;
  }

  try
  {
    *value = values.at<VALUE>(key);
    return true;
  }
  catch (const gtsam::ValuesKeyDoesNotExist& e)
  {
    return false;
  }
}

using gtsam::symbol_shorthand::M;  // dynamic points
using gtsam::symbol_shorthand::P;  // static points
using gtsam::symbol_shorthand::X;  // camera pose

constexpr static char kObjectPoseKey = 'l';
constexpr static char kObjectMotionKey = 'H';

inline gtsam::Key L(unsigned char label, std::uint64_t j)
{
  return gtsam::LabeledSymbol(kObjectPoseKey, label, j);
}
inline gtsam::Key H(unsigned char label, std::uint64_t j)
{
  return gtsam::LabeledSymbol(kObjectMotionKey, label, j);
}

std::string ObjectCentricKeyFormatter(gtsam::Key);

/**
 * @brief Unique gtsam::Key for an object pose based on the object label and frame of the label. This assumes
 * that no two objects in a frame have the same label
 *
 * @param object_label
 * @param frame_id
 * @return gtsam::Key
 */
inline gtsam::Key ObjectPoseKey(int object_label, size_t frame_id)
{
  unsigned char label = object_label + '0';
  return L(label, static_cast<std::uint64_t>(frame_id));
}

inline gtsam::Key ObjectMotionKey(int object_label, size_t frame_id)
{
  unsigned char label = object_label + '0';
  return H(label, static_cast<std::uint64_t>(frame_id));
}

gtsam::io::ValuesFactorGraphPair constructDynaSLAMLikeGraph(const GraphDeconstruction& graph_deconstruction,
                                                            const ObjectLabelResultMap& gt_objects,
                                                            const gtsam::NonlinearFactorGraph& world_centric_graph,
                                                            const gtsam::Values& world_centric_values,
                                                            ObjectManager& object_manager_object_centric,
                                                            object_centric::ObjectIDManager& object_id_manager,
                                                            WorldObjectCentricLikeMapping& world_centric_slam_to_dyna_keys);


std::pair<CameraPoseResultMap, ObjectLabelResultMap> constructDynaLikeResults(
    const CameraPoseResultMap& world_centric_camera_result_map, const ObjectLabelResultMap& world_centric_object_motion_result_map,
    const ObjectManager& object_manager_object_centric, const object_centric::ObjectIDManager& object_id_manager,
    const WorldObjectCentricLikeMapping& world_centric_slam_to_dyna_keys, const gtsam::Values& object_centric_initial_values,
    const gtsam::Values& object_centric_opt_values);