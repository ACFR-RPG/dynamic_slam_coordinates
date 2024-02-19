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

#include "Common.h"
#include <set>

bool Feature::isDynamic() const
{
  return false;
}
bool DynamicFeature::isDynamic() const
{
  return is_dynamic_ && object_label.is_initialized();
}
bool DynamicFeature::isLastInTracklet() const
{
  return isDynamic() && !motion_key.is_initialized();
}

bool StaticFeature::getMeasurement(gtsam::Point3& measurement, const gtsam::Key cam_pose_key) const
{
  for (size_t i = 0; i < camera_pose_keys.size(); i++)
  {
    if (camera_pose_keys[i] == cam_pose_key)
    {
      measurement = measurements[i];
      return true;
    }
  }
  return false;
}

bool StaticFeature::observingCameraPose(const gtsam::Key cam_pose_key) const
{
  auto result = std::find(begin(camera_pose_keys), end(camera_pose_keys), cam_pose_key);
  return result != camera_pose_keys.end();
}

TrackletMap::TrackletMap(const KeyTrackletIdMap& key_to_tracklet_id, const KeyFeatureMap& key_to_feature)
  : key_to_tracklet_id_(key_to_tracklet_id), key_to_feature_(key_to_feature)
{
  for (const auto& it : key_to_tracklet_id)
  {
    const gtsam::Key key = it.first;
    const size_t tracklet_id = it.second;

    if (this->find(tracklet_id) == this->end())
    {
      // new tracklet
      // 1. check if dynamic. We look at the first one to see if dynamic and assume all the other ones are the same
      Feature::shared_ptr feature = key_to_feature_.at(key);
      bool is_dynamic = feature->is_dynamic_;
      Tracklet tracklet(is_dynamic);
      tracklet.push_back(feature);

      CHECK_EQ(tracklet_id, feature->tracklet_id);
      this->operator[](tracklet_id) = tracklet;
    }
    else
    {
      Tracklet& tracklet = this->at(tracklet_id);
      Feature::shared_ptr feature = key_to_feature_.at(key);
      bool is_dynamic = feature->is_dynamic_;
      CHECK_EQ(is_dynamic, tracklet.is_dynamic_);
      CHECK_EQ(tracklet_id, feature->tracklet_id);
      tracklet.push_back(feature);
    }
  }
}

FrameToKeyMap TrackletMap::reorderToPerFrame() const
{
  std::map<gtsam::Key, std::set<gtsam::Key>> cam_key_to_feature_key;

  for (const auto& it : *this)
  {
    const size_t tracklet_id = it.first;
    const Tracklet& tracklet = it.second;

    for (const auto& feature : tracklet)
    {
      auto static_feature = std::dynamic_pointer_cast<StaticFeature>(feature);
      if (static_feature)
      {
        CHECK_EQ(static_feature->measurements.size(), static_feature->camera_pose_keys.size());
        for (const gtsam::Key cam_pose_key : static_feature->camera_pose_keys)
        {
          if (cam_key_to_feature_key.find(cam_pose_key) == cam_key_to_feature_key.end())
          {
            cam_key_to_feature_key[cam_pose_key] = std::set<gtsam::Key>();
          }
          cam_key_to_feature_key[cam_pose_key].insert(feature->key);
        }
      }

      auto dynamic_feature = std::dynamic_pointer_cast<DynamicFeature>(feature);
      if (dynamic_feature)
      {
        const gtsam::Key cam_pose_key = dynamic_feature->camera_pose_key;

        if (cam_key_to_feature_key.find(cam_pose_key) == cam_key_to_feature_key.end())
        {
          cam_key_to_feature_key[cam_pose_key] = std::set<gtsam::Key>();
        }
        cam_key_to_feature_key[cam_pose_key].insert(feature->key);
      }
    }
  }

  // reconstruct to vector
  FrameToKeyMap frame_to_key;
  for (const auto& it : cam_key_to_feature_key)
  {
    gtsam::KeyVector keys(it.second.begin(), it.second.end());
    frame_to_key[it.first] = keys;
  }

  return frame_to_key;
}

void ObjectObservations::addObjectObservations(int object_id, const gtsam::KeyVector& keys)
{
  for (const gtsam::Key& k : keys)
  {
    addObjectObservation(object_id, k);
  }
}

void ObjectObservations::addObjectObservation(int object_id, gtsam::Key key)
{
  safeBackInsrterToMap(*this, object_id, key);
}

std::vector<int> ObjectObservations::ObservedObjects() const
{
  std::vector<int> object;
  for (const auto& it : *this)
  {
    object.push_back(it.first);
  }
  return object;
}

bool ObjectObservations::objectObserved(int object_id) const
{
  return find(object_id) != end();
}

bool ObjectObservations::remap(int old_key, int new_key)
{
  if (!objectObserved(old_key))
  {
    return false;
  }
  // updaete object id
  auto nodeHandler = this->extract(old_key);
  nodeHandler.key() = new_key;
  this->insert(std::move(nodeHandler));
  return true;
}

bool ObjectObservations::remove(int object_id)
{
  if (!objectObserved(object_id))
  {
    return false;
  }
  this->erase(object_id);
  return true;
}

void ObjectManager::safeAdd(const ObjectObservations& observations)
{
  int frame_id = observations.FrameId();
  CHECK(find(frame_id) == end()) << "Bew observation at existing frame id " << frame_id;
  emplace(frame_id, observations);
}

bool ObjectManager::objectInFrame(int frame_id, int object_id) const
{
  if (find(frame_id) == end())
  {
    return false;
  }
  const ObjectObservations& observation = at(frame_id);
  return observation.objectObserved(object_id);
}

int ObjectManager::numObservations(int object_id) const
{
  // assume ordered map
  int count = 0;
  for (const auto& it : *this)
  {
    const int frame_id = it.first;
    if (objectInFrame(frame_id, object_id))
    {
      count++;
    }
  }

  if (count == 0)
  {
    return -1;
  }
  return count;
}

const gtsam::KeyVector& ObjectManager::getObservation(int frame_id, int object_id) const
{
  if (!objectInFrame(frame_id, object_id))
  {
    throw std::runtime_error("Object requested at frame id " + std::to_string(frame_id) + " and object id " +
                             std::to_string(object_id) + " does not exist");
  }

  const ObjectObservations& observation = at(frame_id);
  return observation.at(object_id);
}

int ObjectManager::getClosestPreviousFrame(int frame_id, int object_id) const
{
  CHECK(objectInFrame(frame_id, object_id));

  // iterate backwards
  int previous_frame_id = frame_id - 1;
  for (; previous_frame_id >= 0; previous_frame_id--)
  {
    if (objectInFrame(previous_frame_id, object_id))
    {
      return previous_frame_id;
    }
  }
  return -1;
}

int ObjectManager::getFirstSeenFrame(int object_id) const
{
  // assume ordered map
  for (const auto& it : *this)
  {
    const int frame_id = it.first;
    if (objectInFrame(frame_id, object_id))
    {
      return frame_id;
    }
  }
  return -1;
}
