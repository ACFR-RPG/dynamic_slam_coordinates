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
#include <vector>
#include <gtsam/geometry/Pose3.h>
#include <boost/optional.hpp>

#include <boost/serialization/access.hpp>
#include <boost/serialization/optional.hpp>
#include <map>

#include <memory>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>



#define DSC_POINTER_TYPEDEFS(TypeName)                                                                                 \
  typedef boost::shared_ptr<TypeName> Ptr;                                                                             \
  typedef boost::shared_ptr<const TypeName> ConstPtr;                                                                  \
  typedef std::unique_ptr<TypeName> UniquePtr;                                                                         \
  typedef std::unique_ptr<const TypeName> ConstUniquePtr;                                                              \
  typedef boost::weak_ptr<TypeName> WeakPtr;                                                                           \
  typedef boost::weak_ptr<const TypeName> WeakConstPtr;

#define DSC_DELETE_COPY_CONSTRUCTORS(TypeName)                                                                         \
  TypeName(const TypeName&) = delete;                                                                                  \
  void operator=(const TypeName&) = delete

namespace dsc
{   
using Timestamp = double;
using Depth = double;
using InstanceLabel = int;
using ObjectLabel = int;
using Landmark = gtsam::Point3;
using TrackletId = size_t;

using KeypointCV = cv::KeyPoint;
using KeypointsCV = std::vector<KeypointCV>;
using Landmarks = std::vector<Landmark>;
using Depths = std::vector<Depth>;
using TrackletIds = std::vector<TrackletId>;
using ObjectLabels = std::vector<ObjectLabel>;

using TackletIdToLandmark = std::map<TrackletId, Landmark>;

template <typename T>
using VectorsXx = std::vector<std::vector<T>>;

// double spexalisation
using VectorsXd = VectorsXx<double>;
// int specalisation
using VectorsXi = VectorsXx<int>;


template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}


// assume we have begin and end methods for enahanced loops
template <typename Container>
std::string containerAsString(const Container& container)
{
  std::stringstream ss;
  for (const auto& v : container)
  {
    ss << v << " ";
  }
  return ss.str();
}

/**
 * Numerically stable function for comparing if floating point values are equal
 * within epsilon tolerance.
 * Used for vector and matrix comparison with C++11 compatible functions.
 *
 * If either value is NaN or Inf, we check for both values to be NaN or Inf
 * respectively for the comparison to be true.
 * If one is NaN/Inf and the other is not, returns false.
 *
 * @param check_relative_also is a flag which toggles additional checking for
 * relative error. This means that if either the absolute error or the relative
 * error is within the tolerance, the result will be true.
 * By default, the flag is true.
 *
 * Return true if two numbers are close wrt tol.
 *
 *  References:
 * 1. https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
 * 2. https://floating-point-gui.de/errors/comparison/
 */
bool fp_equal(double a, double b, double tol = 1e-9, bool check_relative_also = true);

template <typename T>
bool equals(const T& a, const T& b, double tol = 1e-9);

inline bool assert_zero(double a)
{
  return fp_equal(a, 0.0);
}

template <typename T>
inline bool equal_with_abs_tol(const std::vector<T>& vec1, const std::vector<T>& vec2, double tol = 1e-9)
{
  if (vec1.size() != vec2.size())
    return false;
  size_t m = vec1.size();
  for (size_t i = 0; i < m; ++i)
  {
    if (!fp_equal(vec1[i], vec2[i], tol))
      return false;
  }
  return true;
}

// Add way of printing strongly typed enums (enum class).
template <typename E>
constexpr typename std::underlying_type<E>::type to_underlying(E e) noexcept
{
  return static_cast<typename std::underlying_type<E>::type>(e);
}

enum class State
{
  kBoostrap,
  kNominal
};

struct ImagePacket
{
  cv::Mat rgb;  // RGB (CV_8UC3) or grayscale (CV_8U)
  cv::Mat depth;
  cv::Mat flow;  // Float (CV_32F).
  cv::Mat semantic_mask;

  ImagePacket()
  {
  }
  ImagePacket(const cv::Mat& rgb_, const cv::Mat& depth_, const cv::Mat& flow_, const cv::Mat& semantic_mask_)
    : rgb(rgb_.clone()), depth(depth_.clone()), flow(flow_.clone()), semantic_mask(semantic_mask_.clone())
  {
  }
};

struct InputPacket
{
  Timestamp timestamp;
  size_t frame_id;
  ImagePacket images;

  InputPacket()
  {
  }

  InputPacket(const Timestamp timestamp_, const size_t frame_id_, const cv::Mat& rgb_, const cv::Mat& depth_,
              const cv::Mat& flow_, const cv::Mat& semantic_mask_)
    : timestamp(timestamp_), frame_id(frame_id_), images(rgb_, depth_, flow_, semantic_mask_)
  {
  }
};

enum class DistortionModel
{
  NONE,
  RADTAN,
  EQUIDISTANT,
  FISH_EYE
};
inline std::string distortionToString(const DistortionModel& model)
{
  switch (model)
  {
    case DistortionModel::EQUIDISTANT:
      return "equidistant";
    case DistortionModel::FISH_EYE:
      return "fish_eye";
    case DistortionModel::NONE:
      return "none";
    case DistortionModel::RADTAN:
      return "radtan";
    default:
      break;
  }
}

struct ObjectPoseGT
{
  size_t frame_id;
  ObjectLabel object_id;

  struct BoundingBox
  {
    int b1 = 0, b2 = 0, b3 = 0, b4 = 0;  // x1, y1, x2, y2 -> starting bottom left corner then top right corner
  };

  BoundingBox bounding_box;
  gtsam::Pose3 pose;  // in the camera frame. To get in world frame X_wc * pose

  template <class Archive>
  void serialize(Archive& ar, const unsigned int)
  {
    ar& BOOST_SERIALIZATION_NVP(frame_id);
    ar& BOOST_SERIALIZATION_NVP(object_id);
    // ar& BOOST_SERIALIZATION_NVP(bounding_box);
    ar& BOOST_SERIALIZATION_NVP(pose);
  }
};

using ObjectPoseGTOptional = boost::optional<ObjectPoseGT>;

struct GroundTruthInputPacket
{
  using ConstOptional = boost::optional<GroundTruthInputPacket>;
  using Optional = boost::optional<GroundTruthInputPacket&>;

  gtsam::Pose3 X_wc;  // pose of the cmaera in the world frame
  std::vector<ObjectPoseGT> obj_poses;
  Timestamp timestamp;
  size_t frame_id;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int)
  {
    ar& BOOST_SERIALIZATION_NVP(X_wc);
    ar& BOOST_SERIALIZATION_NVP(obj_poses);
    ar& BOOST_SERIALIZATION_NVP(timestamp);
    ar& BOOST_SERIALIZATION_NVP(frame_id);
  }

  /**
   * @brief Given a ground truth input packet (other) from another frame and a query object label,
   * try and find the corresponding ground truth object poses.
   *
   * Returns true if the query object was found in this and other.
   *
   * @param other const GroundTruthInputPacket& other ground truth packet to query
   * @param label ObjectLabel object to search for in this and other
   * @param obj ObjectPoseGT& output object ground truth to set from this
   * @param other_obj ObjectPoseGT& output object ground truth to set from other
   * @return true
   * @return false
   */
  inline bool getObjectGTPair(const GroundTruthInputPacket& other, ObjectLabel label, ObjectPoseGT& obj,
                              ObjectPoseGT& other_obj) const
  {
    // is in both object vectors
    bool found = false;
    bool found_other = false;
    for (const auto& o : obj_poses)
    {
      if (o.object_id == label)
      {
        obj = o;
        found = true;
        break;
      }
    }

    for (const auto& o : other.obj_poses)
    {
      if (o.object_id == label)
      {
        other_obj = o;
        found_other = true;
        break;
      }
    }

    return found && found_other;
  }
};

}  // namespace dsc
