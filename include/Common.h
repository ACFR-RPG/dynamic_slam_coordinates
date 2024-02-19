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

#include <vector>

#include "Types.h"
#include "UtilsGtsam.h"
#include "GtsamIO.h"
#include "Metrics.h"
#include "DataProvider.h"

#include <boost/optional.hpp>
#include <memory>

void world_centricSamParse(int argc, char** argv);

void runExperiments();

template <typename MAP>
using MapKey = typename MAP::key_type;

template <typename MAP>
using MapValue = typename MAP::mapped_type;

// assuming mapped value is a container with T::value_type as with vector and set. Could do some smart SFINAE here but
// meh
template <typename MAP>
using MapValueType = typename MapValue<MAP>::value_type;

// only works on map of containers which contain a push_back method
template <typename MAP>
bool safeBackInsrterToMap(MAP& map, const MapKey<MAP>& key, const MapValueType<MAP>& value)
{
  bool is_new = false;
  if (map.find(key) == map.end())
  {
    is_new = true;
    map[key] = MapValue<MAP>();
  }

  map[key].push_back(value);
  return is_new;
}

template <typename CONTAINER>
std::string container_to_string(const CONTAINER& container)
{
  std::stringstream ss;
  for (const auto& v : container)
  {
    ss << v << " ";
  }
  return ss.str();
}

struct Feature
{
  using shared_ptr = std::shared_ptr<Feature>;

  gtsam::Key key;  //! values key in world_centric graph of the 3d point
  size_t tracklet_id;
  bool is_dynamic_{ false };

  virtual ~Feature() = default;
  virtual bool isDynamic() const;
};

struct StaticFeature : public Feature
{
  using shared_ptr = std::shared_ptr<StaticFeature>;
  gtsam::Point3Vector measurements;   //! as defined by the point3d factor. When static should be a list
  gtsam::KeyVector camera_pose_keys;  //! values key in world_centric graph of the observing camera pose

  bool getMeasurement(gtsam::Point3& measurement, const gtsam::Key cam_pose_key) const;
  // true if the camera pose is observed
  bool observingCameraPose(const gtsam::Key cam_pose_key) const;
};

struct DynamicFeature : public Feature
{
  using shared_ptr = std::shared_ptr<DynamicFeature>;

  gtsam::Point3 measurement;
  gtsam::Key camera_pose_key;  // values key in world_centric graph of the observing camera pose

  boost::optional<gtsam::Key> motion_key{ boost::none };  // values key in world_centric graph of attached motion if the feature
                                                          // is dynamic. Should take us from previous pose/point to
                                                          // current pose/point (ie does not propogate this point
                                                          // forward)
  boost::optional<int> object_label{ boost::none };

  bool isDynamic() const override;
  bool isLastInTracklet() const;
};

class Tracklet : public std::vector<Feature::shared_ptr>
{
public:
  Tracklet(bool is_dynamic) : is_dynamic_(is_dynamic)
  {
  }
  Tracklet()
  {
  }

  inline int index(const Feature::shared_ptr& f) const
  {
    auto it = std::find(begin(), end(), f);
    // If element was found
    if (it != end())
    {
      return it - begin();
    }
    else
    {
      return -1;
    }
  }

  bool is_dynamic_{ false };
};

using KeyTrackletIdMap = std::map<gtsam::Key, size_t>;
using KeyFeatureMap = std::map<gtsam::Key, Feature::shared_ptr>;

// this is actually camera pose idx to std::vector (where the key vector contains the keys of each feature)
// but becuase we have a camera pose per frame
// we can get the frame from this anyway
using FrameToKeyMap = std::map<gtsam::Key, gtsam::KeyVector>;

// tracklet Id -> tracklet
class TrackletMap : public std::map<size_t, Tracklet>
{
public:
  TrackletMap(const KeyTrackletIdMap& key_to_tracklet_id, const KeyFeatureMap& key_to_feature);
  TrackletMap()
  {
  }

  inline bool hasFeature(gtsam::Key key) const
  {
    return key_to_feature_.find(key) != key_to_feature_.end();
  }

  inline Feature::shared_ptr atFeature(gtsam::Key key) const
  {
    if (key_to_feature_.find(key) == key_to_feature_.end())
    {
      throw std::runtime_error("Tracklet map missing feature at key " + std::to_string(key));
    }
    return key_to_feature_.at(key);
  }

  inline const Tracklet& trackletFromKey(gtsam::Key key) const
  {
    const Feature::shared_ptr& feature = this->atFeature(key);
    return this->at(feature->tracklet_id);
  }

  // this should be ordered by frame and then by key in increasing order (since we add keys one at a time)
  // allowing us to iterate over the map/vector as if we are getting data new
  FrameToKeyMap reorderToPerFrame() const;

  KeyTrackletIdMap key_to_tracklet_id_;
  KeyFeatureMap key_to_feature_;
};

// object id's to keys for each object (which can be used to access the features via tracklet map)
class ObjectObservations : public std::map<int, gtsam::KeyVector>
{
public:
  ObjectObservations(int frame_id) : frame_id_(frame_id)
  {
  }

  void addObjectObservations(int object_id, const gtsam::KeyVector& keys);
  void addObjectObservation(int object_id, gtsam::Key key);

  // which objects were seen in this frame (keys of the map)
  std::vector<int> ObservedObjects() const;
  // true if the object is present in these observations
  bool objectObserved(int object_id) const;

  bool remap(int old_key, int new_key);

  bool remove(int object_id);

  inline int FrameId() const
  {
    return frame_id_;
  }

private:
  int frame_id_{ -1 };
};

// frame id to object observations
class ObjectManager : public std::map<int, ObjectObservations>
{
public:
  void safeAdd(const ObjectObservations& observations);

  bool objectInFrame(int frame_id, int object_id) const;

  // how many times an object has appeared -1 if not
  int numObservations(int object_id) const;

  const gtsam::KeyVector& getObservation(int frame_id, int object_id) const;

  // the latest previous frame that the object was obsrved in. In most cases should be
  // frame_id - 1, unless the object was dropped. -1 on not found (which is the case if this is the
  // first time the object is seen)
  int getClosestPreviousFrame(int frame_id, int object_id) const;

  //-1 if not seen yet
  int getFirstSeenFrame(int object_id) const;
};

struct GraphDeconstruction
{
  std::vector<size_t> between_factor_idx;
  std::vector<size_t> dynamic_3d_point_idx;
  std::vector<size_t> static_3d_point_idx;
  std::vector<size_t> landmark_ternary_factor_idx;
  std::vector<size_t> smoothing_factor_idx;

  gtsam::KeySet camera_pose_keys;
  gtsam::KeySet object_motion_keys;

  TrackletMap tracklets;
};

enum ReferenceFrame
{
  WORLD,
  CAMERA_CURRENT,   // t = current
  CAMERA_PREVIOUS,  // t-1 = previous
  OBJECT_PREVIOUS,
  OBJECT_CURRENT  // t
};

struct ObjectMotionResult
{
  using shared_ptr = std::shared_ptr<ObjectMotionResult>;
  gtsam::Pose3 ground_truth_L;      // motion
  gtsam::Pose3 ground_truth_W;      // motion
  gtsam::Pose3 ground_truth_C;      // motion in X_{t-1}
  gtsam::Pose3 initial_estimate;    // motion
  gtsam::Pose3 optimized_estimate;  // motion
  gtsam::Pose3 first_frame_camera_pose_gt;
  gtsam::Pose3 second_frame_camera_pose_gt;
  boost::optional<gtsam::Pose3> initial_estimate_L;    // should correspond with object_pose_second_world_gt
  boost::optional<gtsam::Pose3> optimized_estimate_L;  // should correspond with object_pose_second_world_gt (?)
  dsc::ObjectPoseGT::BoundingBox first_frame_bounding_box;
  dsc::ObjectPoseGT::BoundingBox second_frame_bounding_box;
  gtsam::Pose3 object_pose_first_world_gt;
  gtsam::Pose3 object_pose_second_world_gt;
  bool has_estimated = false;

  int object_label;  // gt
  int frame_id;      // of the previous (first) frame
};

// find based on frame id
struct FindFrameIdObjectMotionResult
{
  int frame_id_;
  FindFrameIdObjectMotionResult(int frame_id) : frame_id_(frame_id)
  {
  }
  bool operator()(const ObjectMotionResult& m) const
  {
    return m.frame_id == frame_id_;
  }

  bool operator()(const ObjectMotionResult::shared_ptr& m) const
  {
    return m->frame_id == frame_id_;
  }
};

// per object
using ObjectMotionResults = std::vector<ObjectMotionResult::shared_ptr>;
// object id to results
using ObjectLabelResultMap = std::map<int, ObjectMotionResults>;
// frame id to object motion
using ObjectFrameIdResultMap = std::map<int, ObjectMotionResults>;

struct CameraPoseResult
{
  gtsam::Pose3 ground_truth;  // for now all in world
  gtsam::Pose3 initial_estimate;
  gtsam::Pose3 optimized_estimate;
};
// frame id to camera pose result
using CameraPoseResultMap = std::map<int, CameraPoseResult>;

template <typename Container_>
std::pair<typename Container_::value_type, int> getMostFrequentElement(const Container_& container)
{
  using Value = typename Container_::value_type;
  std::unordered_map<Value, int> elements;
  for (const Value& v : container)
  {
    elements[v]++;
  }
  int maxCount = 0;
  Value res;
  for (auto i : elements)
  {
    if (maxCount < i.second)
    {
      res = i.first;
      maxCount = i.second;
    }
  }
  return std::make_pair(res, maxCount);
}
